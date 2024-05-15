import SimpleITK as sitk
import sys
import os


def main(args):
    if len(args) < 2:
        print(
            "Usage: N4BiasFieldCorrection inputImage "
            + "outputImage [shrinkFactor] [maskImage] [numberOfIterations] "
            + "[numberOfFittingLevels]"
        )
        sys.exit(1)

    inputImage = sitk.ReadImage(args[1], sitk.sitkFloat32)
    image = inputImage

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    shrinkFactor = 1
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    sitk.WriteImage(corrected_image_full_resolution, args[2])

    if shrinkFactor > 1:
        sitk.WriteImage(
            corrected_image, "Python-Example-N4BiasFieldCorrection-shrunk.nrrd"
        )

    return_images = {
        "input_image": inputImage,
        "mask_image": maskImage,
        "log_bias_field": log_bias_field,
        "corrected_image": corrected_image,
    }
    return return_images


if __name__ == "__main__":
    main(sys.argv)
