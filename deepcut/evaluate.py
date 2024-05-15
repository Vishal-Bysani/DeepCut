from cnn import CNN
from deepcut import generate_patches, deepcut
from deepcut import predict, get_probability_map

import denseCRF
import numpy as np
import csv
import os
import json
import argparse
import cv2 as cv
import torch

def manual_segmentation(image_path):
    image_name = os.path.basename(image_path)
    image =  cv.imread('./data/manual_segmentation/' + image_name, cv.IMREAD_GRAYSCALE)
    
    mask = np.zeros((image.shape[0], image.shape[1]))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 255:
                mask[y, x] = 1
    return mask


def evaluate():
    parser = argparse.ArgumentParser('DeepCut')
    parser.add_argument('--cpu', type=int,
                        help='1 if you want to force cpu',
                        default=0)
    parser.add_argument('--data', type=str,
                        help='path of the directory containing the dataset',
                        default="./data")
    parser.add_argument('--batch_size', type=int,
                        help='number of samples used in a mini-batch',
                        default=8192)
    parser.add_argument('--model', type=str,
                        help='existing model weights for inference',
                        default="./models/deepcut.pth")
    parser.add_argument('--image', type=str,
                        help='image for inference',
                        default="image-1.png")
    parser.add_argument('--crf_params', type=str,
                        help='path of the json file containing crf parameters',
                        default='./data/crf_params.json')
    args = parser.parse_args()

    crf_params = json.load(open(args.crf_params, 'r'))
    crf_params = (
        crf_params['w1'],
        crf_params['alpha'],
        crf_params['beta'],
        crf_params['w2'],
        crf_params['gamma'],
        crf_params['it']
    )

    device = 'cpu'
    if args.cpu == 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device).manual_seed(2**31 - 1)
    model = CNN(device=device, generator=generator)
    print(f'pytorch is using {device}')
    generator = torch.Generator(device=device).manual_seed(2**31 - 1)
    image = cv.imread(os.path.join(args.data, f'images/{args.image}'))
    model.load_state_dict(torch.load(args.model))
    model.eval()
    manual_labels = manual_segmentation(args.image)
    batch_size = 8192   # TODO: take as argument

    with open(os.path.join(args.data, f"annotations/{args.image[:-4]}.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            tmp = [int(num) for num in row]
    bb = tmp[:4]
    rgb_image = image.copy()
    image, labels = deepcut(image, bb, model, crf_params, batch_size, device)

    model_foreground = 0
    manual_foreground = 0
    intersection = 0
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] == 1:
                model_foreground += 1
            if manual_labels[y, x] == 1:
                manual_foreground += 1
            if labels[y, x] == 1 and manual_labels[y, x] == 1:
                intersection += 1

    DSC = 2 * intersection / (model_foreground + manual_foreground)
    print(f'Dice Similarity Coefficient: {DSC}')


if __name__ == '__main__':
    evaluate()
