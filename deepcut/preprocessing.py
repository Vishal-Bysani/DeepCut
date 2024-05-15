import os
import csv

import numpy as np
import cv2 as cv


def pre_process(data_dir='data'):
    """
    For image `data_dir/images/img`, annotation is read from
    `data_dir/annotations/img.csv`. It is assumed that the annotation
    contains the coordinates of the bounding box in the format
        x_{top-left},y_{top-left},x_{bottom-right},y_{bottom-right}

    For each pixel in the image, if it is inside the bounding box, it is
    appended to the foreground list, otherwise to the background list.
    These lists are saved as numpy arrays in `data_dir/foreground.npy`
    and `data_dir/background.npy`.

    The two lists contain tuples of the form (img_name, x, y), where
    img_name is the name of the image file, and x, y are the pixel
    coordinates.

    :param data_dir: Directory containing the dataset
    """
    annotations = list(sorted(os.listdir(os.path.join(data_dir, "annotations"))))

    foreground = []  # (img_name, x, y)
    background = []

    for annotation in annotations:
        tmp = []
        with open(os.path.join(data_dir, f"annotations/{annotation}"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                tmp = [int(num) for num in row]
        bb = tmp[:4]

        img = cv.imread(os.path.join(data_dir, "images", annotation[:-4] + ".png"),
                        cv.IMREAD_GRAYSCALE)
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if x < bb[0] or x > bb[2] or y < bb[1] or y > bb[3]:
                    background.append([annotation[:-4], x, y])
                else:
                    foreground.append([annotation[:-4], x, y])

    np.save(os.path.join(data_dir, "foreground.npy"), np.array(foreground, dtype=np.str_))
    np.save(os.path.join(data_dir, "background.npy"), np.array(background, dtype=np.str_))


if __name__ == '__main__':
    pre_process()
