import torch
import torchvision.transforms as T


class Normalize:
    def __call__(self, img):
        return (img - torch.mean(img)) / (torch.std(img) + 1e-6)


class GaussianNoise:
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.randn(image.shape, device=image.device) * self.std + self.mean
        return image + noise


def get_initial_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Grayscale(),
        Normalize(),
        T.Pad(16, padding_mode='reflect')
    ])


def get_transforms(train):
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            GaussianNoise()
        ])
