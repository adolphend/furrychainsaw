import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def heatmapfile(data,  figpath, cmap=None):
    plt.figure()
    sns.heatmap(data, cmap=cmap)
    plt.show()
    plt.savefig(figpath)

def generateFxFy(dimX, dimY, DeltaX, DeltaY):
    y, x = np.indices((dimX, dimY))
    x, y = (x - int(dimX/2)) * DeltaX, (y - (dimY/2)) * DeltaY
    return x, y
