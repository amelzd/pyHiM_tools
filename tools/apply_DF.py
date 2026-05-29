import argparse
import SimpleITK as sitk
import numpy as np
import os


def read_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        arr = np.load(file_path)
        return arr

    elif ext in [".tif", ".tiff"]:
        img = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(img)

    else:
        raise ValueError(f"Unsupported image format: {ext}")


def save_npy(array, file_path):
    np.save(file_path, array.astype(np.float32))


def save_tiff(array, file_path):
    img = sitk.GetImageFromArray(array.astype(np.float32))
    sitk.WriteImage(img, file_path)




def sitk_warp(image_np, df_np):

    image = sitk.GetImageFromArray(image_np.astype(np.float32))

    # deformation field must be vector image
    field = sitk.GetImageFromArray(
        df_np.astype(np.float32),
        isVector=True
    )

    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(sitk.sitkLinear)

    warped = warp_filter.Execute(image, field)

    return sitk.GetArrayFromImage(warped)



def main():

    parser = argparse.ArgumentParser( description="Apply deformation field to 3D image.")
    parser.add_argument( "--image",required=True, help="Input moving image (.tif/.tiff/.npy)")
    parser.add_argument("--DF",required=True,help="Deformation field (.tif/.tiff/.npy)")
    parser.add_argument("--out",required=True, help="Output moved image (.npy)")
    parser.add_argument("--out_tiff",default=None,help="Optional output TIFF path")

    args = parser.parse_args()

    print("Loading image...")
    moving_np = read_image(args.image)

    print("Loading deformation field...")
    field_np = read_image(args.DF)

    print("Warping image...")
    moved_image_np = sitk_warp(moving_np, field_np)

    print(f"Saving NPY: {args.out}")
    save_npy(moved_image_np, args.out)

    if args.out_tiff is not None:
        print(f"Saving TIFF: {args.out_tiff}")
        save_tiff(moved_image_np, args.out_tiff)

    print("Done.")


if __name__ == "__main__":
    main()
