import torch
import re
import glob
import os
import collections
import string
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def smoonth_line(line, every):
    line_s = line.copy()
    total_points = len(line)
    for start_index in range(0, total_points, every):
        # 不能超过最后一个点
        end_index = min(start_index + every, total_points - 1)
        # 获取起始和终止值
        start_value = line[start_index]
        end_value = line[end_index]
        for j in range(start_index, end_index + 1):
            t = (j - start_index) / (end_index - start_index)
            line_s[j] = (1 - t) * start_value + t * end_value             
    return line_s


def calc_psnr(im1, im2):
    im1 = im1.data.cpu().detach().numpy()
    im1 = im1[0].transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy()
    im2 = im2[0].transpose(1, 2, 0)
    return peak_signal_noise_ratio(im1, im2)


def RGB2gray(rgb, keepdim=False):
    if rgb.size(1) == 3:
        r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray if not keepdim else gray.unsqueeze(1)
    elif rgb.size(1) == 1:
        return rgb[:,0,:,:] if not keepdim else rgb

def calc_ssim(im1, im2):
    im1 = im1.data.cpu().detach().numpy().transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy().transpose(1, 2, 0)

    score = structural_similarity(im1, im2, data_range=1, channel_axis=-1)
    return score

def try_convert2numeric(s):
    try:
        if '.' not in s:
            out = int(s)
        else:
            out = float(s)
        return out
    except:
        return s


def is_number_regex(s):
    # This pattern matches integer and floating point numbers, 判断字符串是否是数字
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))


def de_duplication_by_name(d:dict, prefix_len=5):
    de_duplication = {}
    for k, v in d.items():
        for model_name in v:
            if model_name[:prefix_len] in de_duplication:
                continue
            else:
                de_duplication[model_name[:prefix_len]] = model_name
        d[k] = list(de_duplication.values())
    return d


def match_by_keywords(name, keywords, mode='and'):
    if mode.upper() == "AND":
        for keyword in keywords:
            if keyword.upper() not in name.upper():
                return False
        return True
    else:
        for keyword in keywords:
            if keyword.upper() in name.upper():
                return True
            return False

    
def onehot_to_alphabet(label):
    try:
        assert label.dim() == 2
    except:
        print("hello")
    return ''.join(string.ascii_uppercase[i] for i in torch.nonzero(label)[::,-1].tolist())


def count(x, y, dim=-1):
    res = []
    for i, j in zip(x, y):
        map = torch.zeros((26, 26))
        predict_index = i.argmax(dim=-1) 
        if predict_index.shape != j.shape: # CE
            real_index = j.argmax(dim=-1)
        else: # BCE
            real_index = j
        assert real_index.shape == predict_index.shape
        for k, m in zip(real_index, predict_index):
            map[k, m] += 1
        res.append((predict_index == real_index).sum().item())
        res.append(map)
    return res
    # return [(i.argmax(dim=-1) == j.argmax(dim=-1)).sum().item() for i, j in zip(x, y)]


def normalize(x, mode='L1'):
    if mode == 'L1':
        return x / torch.sum(x, dim=-1, keepdim=True)


def histories_rank(histories:dict[dict[list]], epoch_show=None, compareby=None):
    # histories {modelname1:{hisotry}, modelname2:{history}..}
    # history {0:d_list, 1:d_list, 2:d_list} or {'average': d_list}
    d = {}
    model_name_list = []
    for model_name, history in histories.items():
        history_asr = history_asr_mean(history, epoch_show=epoch_show)
        model_name_list.append(model_name)
        # d[modelname] = history_asr
        for index, asr in history_asr.items():
            d.setdefault(index, [])
            d[index].append(asr)
    
    for index, asr_list in d.items():
        asr_sorted_indices = torch.tensor(asr_list).sort(descending=True)[1].tolist()
        d[index] = [model_name_list[i] for i in asr_sorted_indices] 
        
    if compareby == "name":
        d_no_dup = de_duplication_by_name(d)
    return d_no_dup


def history_asr_mean(history:dict[list], epoch_show=None, d_range=10, best=5):
    d = {}
    for index, data_list in history.items():
        data_list_show = sorted(data_list[:epoch_show][d_range:])[-best:]
        d[index] = float(sum(data_list_show)/len(data_list_show))
    return d


def to_para_dict_list(x, upper=True):
    d = {}
    for k, v in x.items():
        l = list(filter(lambda x: bool(x), map(lambda x: x.replace(' ', ''), v.strip().split(','))))
        if l:
            d[k.upper()] = l
    return d


def extract_model_para_list(model_dict, para_dict_list):
    d = {}
    for model_name in model_dict.keys():
        if model_name in para_dict_list.keys():
            d.update({model_name: para_dict_list[model_name]})
    return d


def paras_to_str(paras):
    if not paras:
        return ''
    else:
        return str(paras)


def str_to_paras(Pstring):
    if not Pstring:
        return None
    assert Pstring.startswith('P(') 
    ret = re.findall("[\(\s](?P<name>.+?)[=]", Pstring) # 非贪婪模式
    P = collections.namedtuple('P', ret)
    return eval(Pstring)


def get_Pstring(pthfilename):
    try:
        return re.findall("@([^#]*?)(#.*)?\.pth", pthfilename)[0][0] # 数据集#前要非贪婪
    except:
        print("Hello")



def get_Pkey(pthfilename, type=dict):
    name_key = pthfilename.split("@")[0].upper().split("_")
    para_key = re.findall("([A-Z]+)=", pthfilename)
    key = list(set(name_key + para_key))
    if type == dict:
        key_d = {}
        key_d.update(zip(key, [1]*len(key)))
        return key_d
    elif type==list:
        return key


def str_to_model_kwargs(string):
    d = {}
    for config in string.split('_'):
        d[config.upper()] = True
    return d

def gen_model_configs(model_configs):
    "提取模型此参数配置"
    d = {}

    for k, v in model_configs.items():
        if v and int(v) == 1:
            if "+" in k:
                module_list = k.upper().split("+")
            elif "_" in k:
                module_list = k.upper().split("_")
            else:
                module_list = [k.upper()]
            d[k] = dict(zip(module_list, len(module_list)*[1]))

    return d



class Accumulator:
    "Accmulate on N variables"
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx, reduction_factor=0):
        return (self.data[idx] / reduction_factor) if reduction_factor else self.data[idx]
    
    def to_dict(self, reduction_factor=0):
        d = {}
        for idx in range(len(self.data)):
            d[idx] = (self.data[idx] / reduction_factor) if reduction_factor else self.data[idx]
        return d


class AccumulatorTensor:
    def __init__(self, n):
        self.data = [torch.zeros((26, 26)).float()] * n
    
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [torch.zeros((26, 26)).float()] * len(self.data)
    
    def __getitem__(self, idx, reduction_factor=0):
        return (self.data[idx] / reduction_factor) if reduction_factor else self.data[idx]
    
    def to_dict(self, reduction_factor=0):
        d = {}
        for idx in range(len(self.data)):
            d[idx] = (self.data[idx] / reduction_factor).tolist() if reduction_factor else self.data[idx].tolist() 
        return d


class History:
    "record historical data"
    def __init__(self, items:list or tuple, name=None):
        self.dict = {}
        self.name = name
        for i in items:
            self.dict[i] = []
    
    def add(self, dict):
        for k, v in dict.items():
            self.dict.setdefault(k, []).append(v)
    
    def reset(self, *args):
        if not args:
            for k in self.dict.keys():
                self.dict[k] = []
        else:
            for k in args:
                if k in self.dict:
                    self.dict[k] = []
    
    def __getitem__(self, idx=''):
        if idx:
            return self.dict.get(idx)
        else:
            return self.dict
    
    def state_dict(self, avg=True):
        if avg:
            avg_list = torch.tensor(list(self.dict.values())).float().mean(dim=0).tolist()  
            return self.dict | {'average':avg_list}
        else:
            return self.dict
    
    
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k != 'average':
                self.dict[k] = v
        return self
    
def filter_files(path, suffix='*.pth', recursive=False, inclusions=[], inlogic='AND', exclusions=[], exlogic='AND'):
    pth_paths_filtered = []
    if recursive:
        pth_paths = glob.glob(os.path.join(path, '**', suffix), recursive=recursive)
    else:
        pth_paths = glob.glob(os.path.join(path, suffix), recursive=recursive)

    # 依次打开pth文件获取History字典，并根据index进行过滤
    # 生成histories = {'filename1':{'average':[0.3,0.2,..]}, 'filename2':{'average':[0.3,0.2,..]}}
    for pth_path in pth_paths: # all([])为True, any([])为False
        basename = os.path.basename(pth_path)
        inclu_res = list(map(lambda x: x in basename or x.upper() in basename or x.lower() in basename, inclusions))
        exclu_res = list(map(lambda x: x in basename or x.upper() in basename or x.lower() in basename, exclusions))
        
        if inclu_res and ((inlogic.upper() == 'AND' and not all(inclu_res)) or (inlogic.upper() == "OR" and not any(inclu_res))):
            continue
        if exclu_res and (exlogic.upper() == 'AND' and all(exclu_res)) or (exlogic.upper() == "OR" and any(exclu_res)):
            continue

        pth_paths_filtered.append(pth_path)
    return pth_paths_filtered


def get_legend(input_string):
    if type(input_string) == str:
        output_string = replace_symbols(input_string)
    elif type(input_string) == list:
        output_string = [replace_symbols(string) for string in input_string]
    return output_string


def remove_quotes(input_string):
    output_string = re.sub(r'[\'\"]', '', input_string)
    return output_string


def replace_symbols(s):
    s = remove_quotes(s)
    if "#" in s:
        s, dataset_name = s.split("#")
        dataset_name = dataset_name.replace('.pth', '')
    else:
        dataset_name = ''

    if "@" in s:
        x, y = s.split("@")
    else:
        x = s.replace('.pth', '')
        y = ''
    x_list = x.split("_")
    y_list = y.replace("P(","").replace(")", "").split(",") if y else ""
    out_str = ""
    y_used_set = set()
    for x in x_list:
        x_upper = x.upper()
        for y in y_list:
            add_symbol = " + " if out_str else ""
            if x_upper in y:
                out_str += add_symbol + y
                y_used_set.add(y)
                break
        else:
            add_symbol = " + " if out_str else ""
            out_str += add_symbol + x_upper
    
    y_unused_set = set(y_list) - y_used_set
    out_str = out_str.replace("DEEPCAPTCHA=CNN", "DEEPCAPTCHA")
    if "DEEPCAPTCHA=LSTM" in out_str:
        out_str = out_str.replace("DEEPCAPTCHA=LSTM", "DEEPCAPTCHA + CRNN")
    for string in y_unused_set:
        out_str += ' + ' + string
    out_str += f" on {dataset_name}" if dataset_name else ""
    return out_str


def convert_characters(input_string):
    input_string = str(input_string)
    if input_string.lower() == 'average':
        return 'Average'
    elif input_string.isdigit():
        num = int(input_string) + 1
        if num % 10 == 1 and num != 11:
            return str(num) + 'st' 
        elif num % 10 == 2 and num != 12:
            return str(num) + 'nd'
        elif num % 10 == 3 and num != 13:
            return str(num) + 'rd'
        else:
            return str(num) + 'th'
    else:
        return input_string.title()


def shuffle_images(images, b=False, c=False, h=True, w=True):
    # images shape: (b, c, h, w)
    b, c, h, w = images.shape

    if not (b or c or h or w):
        print("Warning: No image shuffle!")
        return images

    if h:
        h_permutation = torch.randperm(h)
        shuffled_images = images[:, :, h_permutation, :]
    
    if w:
        w_permutation = torch.randperm(w)
        shuffled_images = shuffled_images[:, :, :, w_permutation]
        
    return shuffled_images


    
if __name__ == '__main__':
    # print(str_to_paras(get_Pstirng("deepcaptcha_filter_stn@P(DEEPCAPTCHA='LSTM').pth")))
    # print(str_to_paras("P(DEEPCAPTCHA='BCE', RES='T0_T1_13_23')"))
    batch_size = 2
    channels = 2
    height = 2
    width = 2
    images = torch.rand(batch_size, channels, height, width)
    shuffled_images = shuffle_images(images)





