import torch.nn as nn
import torch.nn.functional as F
import torch


class ThinPlateSpline(nn.Module):
    def __init__(self, ctrlshape=(6,6)):
        super().__init__()
        self.nctrl = ctrlshape[0] * ctrlshape[1] # 控制点的总个数
        self.nparam = (self.nctrl + 2) # 待求解的总参数量，不包含a0, 因为所有系数之和为0
        ctrl = ThinPlateSpline.uniform_grid(ctrlshape)
        self.register_buffer('ctrl', ctrl.view(-1, 2)) # 将控制点坐标展平

        self.f = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((5,5))
        )

        self.ctrl = torch.nn.Sequential(
            torch.nn.Linear(25*128, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.nctrl*2), # 乘2的原因是既有x的参数又有y的参数
            torch.nn.Sigmoid() # 将输出参数(采样区间)调整到(0,1)
        )

        self.ctrl[-2].weight.data.normal_(0, 1e-3)
        self.ctrl[-2].bias.data.zero_()

        self.loc = torch.nn.Sequential(
            torch.nn.Linear(25*128, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.nparam*2), # 乘2的原因是既有x的参数又有y的参数
            torch.nn.Tanh() # 将输出参数(代求系数)调整到(-inf, inf)
        )   

        self.loc[-2].weight.data.normal_(0, 1e-3)
        self.loc[-2].bias.data.zero_()

    @staticmethod
    def uniform_grid(shape):
        '''Uniformly places control points aranged in grid accross normalized image coordinates.
        
        Params
        ------
        shape : tuple
            HxW defining the number of control points in height and width dimension

        Returns
        -------
        points: HxWx2 tensor
            Control points over [0,1] normalized image range.
        '''
        H, W = shape[:2]
        c = torch.zeros(H, W, 2)
        c[...,0] = torch.linspace(0, 1, W)
        c[...,1] = torch.linspace(0, 1, H).unsqueeze(-1)
        return c

    @staticmethod
    def tps_grid(theta, ctrl, size):
        '''Compute a thin-plate-spline grid from parameters for sampling.
        
        Params
        ------
        theta: Nx(T+3)x2 or Nx(T+2)x2  tensor
            Batch size N, T+3 model parameters for T control points in dx and dy.
        ctrl: NxTx2 tensor, or Tx2 tensor
            T control points in normalized image coordinates [0..1]
        size: tuple
            Output grid size as NxCxHxW. C unused. This defines the output image
            size when sampling.
        
        Returns
        -------
        grid : NxHxWx2 tensor
            Grid suitable for sampling in pytorch containing source image
            locations for each output pixel.
        ''' 
        if len(size) == 4:
            N, _, H, W = size  
        else:
            N, H, W = size
        
        grid = theta.new(N, H, W, 3)
        grid[:, :, :, 0] = 1.
        grid[:, :, :, 1] = torch.linspace(0, 1, W)
        grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)   
        
        z = ThinPlateSpline.tps(theta, ctrl, grid)
        return (grid[...,1:] + z)*2-1 # [-1,1] range required by F.sample_grid

    @staticmethod
    def tps(theta, ctrl, grid):
        '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
        The TPS surface is a minimum bend interpolation surface defined by a set of control points.
        The function value for a x,y location is given by
        
            TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
            
        This method computes the TPS value for multiple batches over multiple grid locations for 2 
        surfaces in one go.
        
        Params
        ------
        theta: Nx(T+3)x2 tensor, or Nx(T+2)x2 tensor
            Batch size N, T+3 or T+2 (reduced form) model parameters for T control points in dx and dy.
        ctrl: NxTx2 tensor or Tx2 tensor
            T control points in normalized image coordinates [0..1]
        grid: NxHxWx3 tensor
            Grid locations to evaluate with homogeneous 1 in first coordinate.
            
        Returns
        -------
        z: NxHxWx2 tensor
            Function values at each grid location in dx and dy.
        '''

        N, H, W, _ = grid.size()

        if ctrl.dim() == 2:
            ctrl = ctrl.expand(N, *ctrl.size())
        
        T = ctrl.shape[1]

        diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
        D = torch.sqrt((diff**2).sum(-1))
        U = (D**2) * torch.log(D + 1e-6)

        w, a = theta[:,:-3,:], theta[:,-3:,:]

        reduced = T + 2  == theta.shape[1]
        if reduced:
            w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1)
        
        # U is NxHxWxT
        b = torch.bmm(U.view(N, -1, T), w).view(N,H,W,2)
        # b is NxHxWx2
        z = torch.bmm(grid.view(N,-1,3), a).view(N,H,W,2) + b
        
        return z


    def forward(self, x):
        xs = self.f(x)
        theta = self.loc(xs.view(x.shape[0], -1)).view(-1, self.nparam, 2)
        ctrlp = self.ctrl(xs.view(x.shape[0], -1)).view(-1, self.nctrl, 2)
        grid = ThinPlateSpline.tps_grid(theta, ctrlp, tuple(x.shape))
        return F.grid_sample(x, grid), theta 