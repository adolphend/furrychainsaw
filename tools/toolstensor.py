import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def heatmapfile(data,  figpath, cmap=None):
    plt.figure()
    sns.heatmap(data, cmap=cmap)
    plt.show()
    plt.savefig(figpath)

def generateFxFy(dimX, dimY, DeltaX, DeltaY):
    y, x = np.indices((dimX, dimY))
    y, x = torch.from_numpy(y).to(device), torch.from_numpy(x).to(device)
    x, y = (x - int(dimX/2)) * DeltaX, (y - (dimY/2)) * DeltaY
    return x, y
