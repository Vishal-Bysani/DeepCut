from cnn import CNN
from deepcut import train, deepcut
from data import Dataset
from transforms import get_transforms
from preprocessing import pre_process

import os
import csv
import json
import argparse
import cv2 as cv
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as figure
from matplotlib.axes._axes import Axes as axes
mpl.use("pgf")


def main():
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
    parser.add_argument('--train', type=int,
                        help='1 if you want to train, 0 otherwise',
                        default=0)
    parser.add_argument('--model', type=str,
                        help='existing model weights for inference',
                        default="./models/deepcut.pth")
    parser.add_argument('--image', type=str,
                        help='image for inference',
                        default="image-1.png")
    parser.add_argument('--crf_params', type=str,
                        help='path of the json file containing crf parameters',
                        default='./data/crf_params.json')
    parser.add_argument('--continue_training', type=int,
                        help='1 if you want to continue training, 0 otherwise',
                        default=0)
    parser.add_argument('--savefig', type=int, help='1 if you want to save the figure',
                        default=0)
    parser.add_argument('--epochs_per_save', type=int,
                        help='number of epochs to save the CNN weights',
                        default=10)
    parser.add_argument('--start_with_crf', type=int,
                        help='1 if you want to start with CRF, 0 otherwise',
                        default=0)
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
    print(f'pytorch is using {device}')
    torch.set_default_device(device)
    generator = torch.Generator(device=device).manual_seed(2**31 - 1)
    model = CNN(device=device, generator=generator)

    if args.train == 0:
        model.load_state_dict(torch.load(args.model))
        model.eval()
        orig_image = cv.imread(str(os.path.join(args.data, "images", args.image)))
        with open(os.path.join(args.data, f"annotations/{args.image[:-4]}.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                tmp = [int(num) for num in row]
        bb = tmp[:4]
        image, labels = deepcut(orig_image, bb, model,
                                crf_params, args.batch_size, device)
        width = 345 / 72.27
        height = 5.6
        fig, axs = plt.subplots(2, 2, figsize=(width, height))    # type: figure, axes
        axs[0, 0].imshow(orig_image)
        axs[0, 0].set_title('Image')
        axs[0, 1].imshow(image)
        axs[0, 1].set_title('Segmentation')
        axs[1, 0].imshow(labels, cmap='gray')
        axs[1, 0].set_title('Labels')
        
        image_path = os.path.join(args.data, "manual_segmentation", args.image)
        if os.path.exists(image_path):
            axs[1, 1].imshow(cv.imread(str(image_path)), cmap='gray')
            axs[1, 1].set_title('Manual labels')
        else:
            print(f"Image file {image_path} does not exist.")
        if args.savefig == 1:
            plt.savefig(f'report/result_{args.image[:-4]}.pgf')
        else:
            plt.show()
    else:
        if args.continue_training == 1:
            model.load_state_dict(torch.load(args.model))
        pre_process(args.data)
        train_dataset = Dataset(args.data, get_transforms(True), device)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, generator=generator)
        losses = train(model, train_loader, train_dataset, crf_params,
                       args.batch_size, device, args.data,
                       num_epochs=500,
                       num_epochs_per_crf=15,
                       num_epochs_first_crf=40,
                       epochs_per_save=args.epochs_per_save,
                       continue_training=args.continue_training)
        fig, ax = plt.subplots(figsize=(12, 3))    # type: figure, axes
        tick_spacing = len(losses) // 50 if len(losses) > 50 else 1
        ax.plot(losses)
        ax.set_title('Loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        dt_string = datetime.now().strftime("%d %H %M").replace(' ', '_')
        ax.set_xticks(range(0, len(losses), tick_spacing))
        plt.savefig(os.path.join(args.data, f'loss_{dt_string}.png'))
        plt.show()


if __name__ == '__main__':
    main()
