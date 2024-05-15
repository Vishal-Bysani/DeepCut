import sys
import numpy as np
import cv2 as cv
import nrrd

def main():
    if len(sys.argv) != 2:
        print('Usage: python converter.py <input.nrrd>')
        sys.exit(1)
    else:
        input_nrrd = sys.argv[1]
    data, header = nrrd.read(input_nrrd)
    data = data.T
    cv.imwrite(input_nrrd[:-5] + '.png', data)


if __name__ == '__main__':
    main()
