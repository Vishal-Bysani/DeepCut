from data import Dataset
from cnn import CNN, train_one_epoch
from transforms import get_initial_transforms

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import denseCRF
import cv2 as cv
import numpy as np
import os
import csv
from datetime import datetime


def generate_patches(rgb_image, patch_size=33) -> torch.tensor:
    """
    :param rgb_image: RGB image (H, W, C=3)
    :param patch_size: size of the patch
    :return: patches of size (N, 1, patch_size, patch_size)
    """
    transforms = get_initial_transforms()
    image = rgb_image.copy()
    image = transforms(image)
    patches = []
    for y in range(0, image.shape[1] - 32, 1):
        for x in range(0, image.shape[2] - 32, 1):
            patch = image[:, y:y + 33, x:x + 33].view(1, 1, 33, 33)
            if patches is None:
                patches = patch
            else:
                patches.append(patch)
    return torch.vstack(patches)


def predict(model, patches, batch_size, image_size, device) -> torch.tensor:
    """
    :param rgb_image: RGB image (H, W, C=3)
    :param batch_size: number of samples used in a mini-batch
    :param device: device to use
    :return: labels for every pixel in an image, shape (H, W)
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    predictions = []
    num_batches = len(patches) // batch_size
    for idx in range(num_batches):
        with torch.no_grad():
            batch = patches[idx * batch_size:(idx + 1) * batch_size]
            batch = batch.to(device)
            prediction = softmax(model(batch))[:, 1].detach()
            predictions.append(prediction)
    predictions = torch.vstack(predictions).flatten()
    if len(patches) % batch_size:
        batch = patches[-(len(patches) % batch_size):]
        batch = batch.to(device)
        predictions = torch.cat((predictions, softmax(model(batch))[:, 1].detach().flatten()))
    return predictions.view(image_size)


def get_probability_map(prediction: torch.tensor) -> np.ndarray:
    """
    :param prediction: labels for every pixel in an image, shape (H, W)
    :return: probability map of shape (H, W, C=2),
             where C is the number of classes
    """
    prob = np.stack((prediction, prediction), axis=-1).astype(np.float32)
    prob[:, :, 0] = 1.0 - prob[:, :, 0]
    return prob


def train(
        model: nn.Module,
        train_loader: DataLoader,
        train_set: Dataset,
        crf_params: tuple[float, float, float, float, float, int],
        batch_size: int,
        device: str,
        data_dir: str = 'data',
        num_epochs: int = 50,
        num_epochs_per_crf: int = 10,
        num_epochs_first_crf: int = 30,
        epochs_per_save: int = 10,
        continue_training: bool = False
) -> list[float]:
    """
    :param device: device to use
    :param num_epochs: Number of epochs till CRF
    :param crf_params: (w1, alpha, beta, w2, gamma, it)
                       w1:    weight of bilateral term, e.g. 10.0
                       alpha: spatial distance std, e.g., 80
                       beta:  rgb value std, e.g., 15
                       w2:    weight of spatial term, e.g., 3.0
                       gamma: spatial distance std for spatial term, e.g., 3
                       it:    iteration number, e.g., 5
    :return: list of losses
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    loss = None
    start_crf = False
    for epoch in range(num_epochs):
        if (epoch + 1) == num_epochs_first_crf:
            start_crf = True
        if not continue_training:
            loss = train_one_epoch(model, optimizer, train_loader, criterion, device)
            losses.append(loss)
        if continue_training or (start_crf and ((epoch + 1) % num_epochs_per_crf == 0)):
            new_foregrounds = []
            new_backgrounds = []
            print('Running denseCRF... ', end='')
            for image_name in train_set.rgb_images.keys():
                bb = train_set.annotations[image_name][:4]
                rgb_image = train_set.rgb_images[image_name]
                prediction = predict(model, train_set.patches[image_name],
                                     batch_size, rgb_image.shape[:2], device).cpu().numpy()
                for y in range(prediction.shape[0]):
                    for x in range(prediction.shape[1]):
                        if not (bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]):
                            prediction[y, x] = 0
                updated_label = torch.where(prediction > 0.5, 1, 0)
                # prob = get_probability_map(prediction)
                # updated_label = denseCRF.densecrf(rgb_image, prob, crf_params)
                for y in range(updated_label.shape[0]):
                    for x in range(updated_label.shape[1]):
                        if (updated_label[y, x] == 1
                                and bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]):
                            new_foregrounds.append([image_name, x, y])
                        else:
                            new_backgrounds.append([image_name, x, y])
            print('Done')
            train_set.foreground = new_foregrounds
            train_set.background = new_backgrounds
            train_set.len = 2 * min(len(new_backgrounds), len(new_foregrounds))
            continue_training = False
        if (epoch + 1) % epochs_per_save == 0:
            dt_string = datetime.now().strftime("%d %H %M").replace(' ', '_')
            torch.save(model.state_dict(),
                       os.path.join(data_dir,
                                    f'models/deepcut_{epoch + 1}_{num_epochs_per_crf}_{dt_string}.pth'))
        if loss is not None:
            print(f'Epoch: {epoch + 1:3d}/{num_epochs:3d}\tLoss: {loss:.5f}')
    return losses


def deepcut(rgb_image, bounding_box, model,
            crf_params, batch_size, device,
            border_color=(0, 255, 0),
            border_thickness=1) -> tuple[np.ndarray, np.ndarray]:
    """
    :param rgb_image: RGB image (H, W, C=3)
    :param bounding_box: (x1, y1, x2, y2) coordinates of the bounding box
    :param model: the cnn
    :param batch_size: number of samples used in a mini-batch
    :param device: device to use
    :param crf_params: (w1, alpha, beta, w2, gamma, it)
                       w1:    weight of bilateral term, e.g. 10.0
                       alpha: spatial distance std, e.g., 80
                       beta:  rgb value std, e.g., 15
                       w2:    weight of spatial term, e.g., 3.0
                       gamma: spatial distance std for spatial term, e.g., 3
                       it:    iteration number, e.g., 5
    :param border_color: color for the border
    :param border_thickness: thickness of the border
    :return: image with border, shape (H, W, C=3), labels, shape (H, W)
    """
    patches = generate_patches(rgb_image)
    prediction = predict(model, patches, batch_size, rgb_image.shape[:2], device).cpu().numpy()
    for y in range(prediction.shape[0]):
        for x in range(prediction.shape[1]):
            if not (bounding_box[0] <= x <= bounding_box[2]
                    and bounding_box[1] <= y <= bounding_box[3]):
                prediction[y, x] = 0
    labels = torch.where(prediction > 0.5, 1, 0).cpu().numpy()
    # prob = get_probability_map(prediction)
    # labels = denseCRF.densecrf(rgb_image, prob, crf_params)
    # for y in range(prediction.shape[0]):
    #     for x in range(prediction.shape[1]):
    #         if (labels[y, x] == 0
    #                 and bounding_box[0] <= x <= bounding_box[2]
    #                 and bounding_box[1] <= y <= bounding_box[3]):
    #             labels[y, x] = 1
    #         else:
    #             labels[y, x] = 0

    mask = labels.astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bordered_image = cv.drawContours(rgb_image.copy(), contours, -1, border_color, border_thickness)

    return bordered_image, labels
