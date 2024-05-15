from transforms import get_initial_transforms

import os
import cv2 as cv
import csv
import numpy as np
import torch.utils.data.dataset


def load_images(dir, rgb=False) -> dict[str, np.ndarray]:
    """
    :param rgb: If True, the images are read as RGB images of dimensions
                (H, W, C=3), otherwise as grayscale images of dimensions
                (C=1, H, W) with padding
    """
    res = {}
    transforms = get_initial_transforms()
    for filename in sorted(os.listdir(dir)):
        val = cv.imread(os.path.join(dir, filename))
        if rgb:
            res[filename[:-4]] = val
        else:
            res[filename[:-4]] = transforms(val)
    return res


def create_patches(rgb_images: dict, patch_size=33) -> dict[str, torch.tensor]:
    """
    TODO: doc comment
    """
    patch_dict = {}
    transforms = get_initial_transforms()
    for name, image in rgb_images.items():
        image = transforms(image)
        patches = []
        for y in range(0, image.shape[1] - 32, 1):
            for x in range(0, image.shape[2] - 32, 1):
                patch = image[:, y:y + patch_size, x:x + patch_size].view(1, 1, 33, 33)
                if patches is None:
                    patches = patch
                else:
                    patches.append(patch)
        patches = torch.vstack(patches)
        patch_dict[name] = patches
    return patch_dict


def load_annotations(dir) -> dict[str, list[int]]:
    """
    :return: a dictionary with image names as keys and lists of annotations as values
    """
    res = {}
    for filename in sorted(os.listdir(dir)):
        with open(os.path.join(dir, filename), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                res[filename[:-4]] = [int(num) for num in row]
    return res


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, transforms, device):
        self.data_dir = data_dir
        self.transforms = transforms
        self.rgb_images = load_images(os.path.join(data_dir, "images"), rgb=True)
        self.patches = create_patches(self.rgb_images)
        self.images = load_images(os.path.join(data_dir, "images"))
        self.annotations = load_annotations(os.path.join(data_dir, "annotations"))
        self.background = np.load(os.path.join(data_dir, "background.npy"))
        self.foreground = np.load(os.path.join(data_dir, "foreground.npy"))
        self.len = 2 * min(self.foreground.shape[0], self.background.shape[0])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx % self.len

        if idx % 2 == 0:
            label = 0
            name = self.background[int(idx / 2)][0]
            x, y = self.background[int(idx / 2)][1:]
            x = int(x)
            y = int(y)
            patch = self.images[name][:, y: y + 33, x: x + 33]
        else:
            label = 1
            name = self.foreground[int((idx - 1) / 2)][0]
            x, y = self.foreground[int((idx - 1) / 2)][1:]
            x = int(x)
            y = int(y)
            patch = self.images[name][:, y: y + 33, x: x + 33]

        return self.transforms(patch), label
