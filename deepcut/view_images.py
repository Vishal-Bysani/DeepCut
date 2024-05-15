from transforms import get_initial_transforms

import os
import sys
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as figure
from matplotlib.axes._axes import Axes as axes


if __name__ == '__main__':
    args = sys.argv
    orig_image = cv.imread(args[1])
    transforms = get_initial_transforms()
    image = orig_image.copy()
    image = transforms(image)
    image = torch.transpose(image, 0, 2)
    image = torch.transpose(image, 0, 1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(image, cmap='gray')
    ax[1].set_title('Transformed Image')
    plt.show()
