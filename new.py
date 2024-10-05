import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib
import matplotlib as mpl
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def confusion_matrix(path=r"D:\show\focal\deepcaptcha_loss@P(DEEPCAPTCHA='LSTM', LOSS='BCEFOCAL-0.1-1').pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # 模拟分类正确率数据
    vegetables = list(string.ascii_uppercase)
    farmers = list(string.ascii_uppercase)

    confusion_matrix = np.round(torch.tensor(checkpoint['train_errormap']['average'][-1]).detach().data.numpy()*4, 0)
    # confusion_matrix = (confusion_matrix / confusion_matrix.sum(dim=-1, keepdims=True)).detach().data.numpy()

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(confusion_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")

        # 调整图表边缘的间距
    plt.tight_layout()
    fig.savefig("./confusion_matrix" , dpi=600, bbox_inches='tight')
    # 显示图表
    plt.show()

# confusion_matrix(path=r"D:\show\focal\新建文件夹 (2)\PCE\deepcaptcha@P(DEEPCAPTCHA='CNN').pth")



import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
  
# 假设A1和A2是已经给定的两个数组，每个数组包含26个值  
np.random.seed(0)  
A1 = np.random.rand(26)  
A2 = np.random.rand(26)  
  
# 计算下降幅度  
difference = A1 - A2  
  
# 创建一个DataFrame来存储数据  
data = pd.DataFrame({  
    'Letter': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),  
    'A1 Value': A1,  
    'A2 Value': A2,  
    'Difference': difference  
})  
  
# 为了瀑布图，我们需要对数据进行一些处理  
# 我们将A1作为起始点，A2作为终点，中间用Difference的负值来表示下降  
data['Start'] = data['A1 Value']  
data['End'] = data['A2 Value']  
data['Decrease'] = -data['Difference']  # 因为瀑布图中下降用负值表示  
  
# 为了在瀑布图中分隔每个字母的变化，我们添加一个索引列  
data['Index'] = range(26)  
  
# 创建瀑布图  
fig, ax = plt.subplots(figsize=(12, 8))  
  
# 绘制起始点（A1 Value）  
ax.bar(data['Index'], data['Start'], color='skyblue', label='A1 Value')  
  
# 在起始点基础上叠加下降幅度（用负数表示下降）  
ax.bar(data['Index'], data['Decrease'], bottom=data['Start'], color='red', label='Decrease')  
  
# 设置图表标题和标签  
ax.set_title('Waterfall Plot Showing Decrease from A1 to A2 Values')  
ax.set_xlabel('Letter Index')  
ax.set_ylabel('Value')  
  
# 添加图例  
ax.legend()  
  
# 为了使图表更炫酷，我们可以添加一些样式  
# 例如，设置网格线样式、坐标轴刻度等  
ax.grid(True, linestyle=':', linewidth=0.5)  
ax.tick_params(axis='x', rotation=45)  # 旋转x轴标签以便更好地显示  
  
# 在每个条形上方添加值标签（可选）  
for i, row in data.iterrows():  
    ax.text(i, row['A2 Value'] + 0.01, f'{row["Letter"]}: {row["A2 Value"]:.2f}', ha='center')  
  
# 显示图表  
plt.tight_layout()  
plt.show()