from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys


def to_uint8(data):
    data -= data.min()
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, output_dir, rgb=False):
    output_dir = Path(output_dir)
    data = nib.load(input_path).get_fdata()
    *_, num_slices, num_channels = data.shape
    for channel in range(num_channels):
        volume = data[..., channel]
        volume = to_uint8(volume)
        channel_dir = output_dir / f'channel_{channel}'
        channel_dir.mkdir(parents=True, exist_ok=True)
        for slice_idx in range(num_slices):
            slice_ = volume[..., slice_idx]
            if rgb:
                slice_ = np.stack([slice_]*3, axis=-1)
            plt.imsave(channel_dir / f'slice_{slice_idx}.jpg', slice_, cmap='gray')


if __name__ == '__main__':
    nii_to_jpgs(sys.argv[1], sys.argv[2], rgb=True)
