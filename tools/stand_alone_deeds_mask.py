import argparse
import SimpleITK as sitk
import numpy as np
import psutil
import h5py
import os, sys
import time
import matplotlib.pyplot as plt
from scipy.ndimage import shift as shift_image
from deeds import registration_imwarp_fields
from skimage import exposure
from photutils.background import MedianBackground
from photutils.background import Background2D
from astropy.stats import SigmaClip
from tqdm import trange
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


##### reading writing and sitk format #####"
def read_image(file_path):
    return sitk.ReadImage(file_path)

def write_image(image, file_path):
    sitk.WriteImage(image, file_path)

def to_numpy(img):
    return sitk.GetArrayFromImage(img)

def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img.astype(np.float32))
    if ref_img:
        img.CopyInformation(ref_img)
    return img

#### pre process image ########
def check_file_existence(reference, moving):
    if not os.path.exists(reference):
        print(f"Error: Reference image file '{reference}' does not exist.")
        sys.exit(1)
    if not os.path.exists(moving):
        print(f"Error: Moving image file '{moving}' does not exist.")
        sys.exit(1)
        
def reinterpolate_z(image_3d, z_range, mode='remove'):
    output = np.zeros((len(z_range), image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype)
    for i, index in enumerate(z_range):
        output[i, :, :] = image_3d[index, :, :]
    return output

def _remove_inhomogeneous_background(im, box_size=(32, 32), filter_size=(3, 3), parallel_execution=True, background=False):
    if len(im.shape) == 2:
        return _remove_inhomogeneous_background_2d(im, filter_size=filter_size, background=background)
    elif len(im.shape) == 3:
        return _remove_inhomogeneous_background_3d(im, box_size=box_size, filter_size=filter_size, parallel_execution=parallel_execution, background=background)
    else:
        return None

def _remove_inhomogeneous_background_2d(im, filter_size=(3, 3), background=False):
    print("Removing inhomogeneous background from 2D image...")
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    bkg = Background2D(im, (64, 64), filter_size=filter_size, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    im1_bkg_substracted = im - bkg.background
    return (im1_bkg_substracted, bkg) if background else im1_bkg_substracted

def _remove_inhomogeneous_background_3d(image_3d, box_size=(64, 64), filter_size=(3, 3), parallel_execution=True, background=False):
    number_planes = image_3d.shape[0]
    output = np.zeros(image_3d.shape)
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    print(f"> Removing inhomogeneous background from {number_planes} planes using 1 worker...")
    z_range = trange(number_planes)
    for z in z_range:
        image_2d = image_3d[z, :, :]
        bkg = Background2D(image_2d, box_size, filter_size=filter_size, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        output[z, :, :] = image_2d - bkg.background
    return (output, bkg.background) if background else output

def image_adjust(image, lower_threshold=0.3, higher_threshold=0.9999):
    image1 = exposure.rescale_intensity(image, out_range=(0, 1))
    hist1_before = exposure.histogram(image1)
    hist_sum = np.zeros(len(hist1_before[0]))
    for i in range(len(hist1_before[0]) - 1):
        hist_sum[i + 1] = hist_sum[i] + hist1_before[0][i]
    sum_normalized = hist_sum / hist_sum.max()
    lower_cutoff = np.where(sum_normalized > lower_threshold)[0][0] / 255
    higher_cutoff = np.where(sum_normalized > higher_threshold)[0][0] / 255
    image1 = exposure.rescale_intensity(image1, in_range=(lower_cutoff, higher_cutoff), out_range=(0, 1))
    hist1 = exposure.histogram(image1)
    return image1, hist1_before, hist1, lower_cutoff, higher_cutoff

def preprocess_3d_image(x, lower_threshold, higher_threshold, parallel_execution=True):
    image = exposure.rescale_intensity(x, out_range=(0, 1))
    image = _remove_inhomogeneous_background(image, parallel_execution=parallel_execution)
    image = image_adjust(image, lower_threshold, higher_threshold)[0]
    return image

def preprocess_images(fixed_image_np, moving_image_np, args):
    print(f"$ Shifting image using: {args.shifts}")
    
    shift_3d = np.zeros((3))
    shift_3d[0], shift_3d[1], shift_3d[2] = args.shifts[2], args.shifts[0], args.shifts[1]
    
    moving_image_np = shift_image(moving_image_np, shift_3d)

    # Apply binning if necessary
    if args.binning_factor_xy > 1 or args.binning_factor_z > 1:
        print(f"$ Applying binning to images: XY binning={args.binning_factor_xy}, Z binning={args.binning_factor_z}")
        zoom_factors = [1/args.binning_factor_z, 1/args.binning_factor_xy, 1/args.binning_factor_xy]
        fixed_image_np = zoom(fixed_image_np, zoom_factors, order=1)
        moving_image_np = zoom(moving_image_np, zoom_factors, order=1)

    print(f"$ Preprocessing images with z_binning={args.z_binning}, lower_threshold={args.lower_threshold}, higher_threshold={args.higher_threshold}")
    fixed_image_np = preprocess_3d_image(fixed_image_np, args.lower_threshold, args.higher_threshold)
    moving_image_np = preprocess_3d_image(moving_image_np, args.lower_threshold, args.higher_threshold)
    
    return fixed_image_np, moving_image_np
    
#### blocks#####
def split_image(image_np, factors):
    z_size = image_np.shape[0]
    y_size = image_np.shape[1]
    x_size = image_np.shape[2]
    block_size = [z_size, y_size // factors[0], x_size // factors[1]]
    blocks = []
    for y in range(factors[0]):
        for x in range(factors[1]):
            y_start = y * block_size[1]
            x_start = x * block_size[2]
            y_end = (y + 1) * block_size[1]
            x_end = (x + 1) * block_size[2]
            if y == factors[0] - 1:
                y_end = y_size
            if x == factors[1] - 1:
                x_end = x_size
            block = image_np[:, y_start:y_end, x_start:x_end]
            blocks.append(block)
    return blocks, block_size, factors

def resample_image(image, new_size, reference_image):
    """
    Resample the given image to a new size based on the reference image.

    Parameters:
    image : SimpleITK.Image
        The input image to be resampled.
    new_size : tuple of int
        The new size for the image (in the format [width, height, depth]).
    reference_image : SimpleITK.Image
        The reference image to copy spacing, origin, and direction from.

    Returns:
    SimpleITK.Image
        The resampled image with the new size.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)

def stitch_blocks(blocks, blocks_shape, block_size, original_shape, is_vector=False):
    if is_vector:
        stitched_image = np.zeros((original_shape[0], original_shape[1], original_shape[2], blocks[0].shape[-1]), dtype=blocks[0].dtype)
    else:
        stitched_image = np.zeros((original_shape[0], original_shape[1], original_shape[2]), dtype=blocks[0].dtype)
    block_index = 0
    for y in range(blocks_shape[0]):
        for x in range(blocks_shape[1]):
            y_start = y * block_size[1]
            x_start = x * block_size[2]
            y_end = (y + 1) * block_size[1]
            x_end = (x + 1) * block_size[2]
            if y == blocks_shape[0] - 1:
                y_end = original_shape[1]
            if x == blocks_shape[1] - 1:
                x_end = original_shape[2]
            if is_vector:
                stitched_image[:, y_start:y_end, x_start:x_end, :] = blocks[block_index]
            else:
                stitched_image[:, y_start:y_end, x_start:x_end] = blocks[block_index]
            block_index += 1
    return stitched_image

############ DF ################"""
def write_displacement_field(displacement_field, file_path, format):
    if format == 'tif':
        sitk.WriteImage(displacement_field, file_path)
    elif format == 'nii':
        sitk.WriteImage(displacement_field, file_path)
    elif format == 'h5':
        displacement_array = sitk.GetArrayFromImage(displacement_field)
        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset('image', data=displacement_array, compression='gzip')
            h5_file.attrs['spacing'] = displacement_field.GetSpacing()
            h5_file.attrs['origin'] = displacement_field.GetOrigin()
            h5_file.attrs['direction'] = displacement_field.GetDirection()
    else:
        raise ValueError("Unsupported file format for displacement field.")


def calculates_deformation(reference, moving, args, method = 'DEEDs'):
    if method=='DEEDs':
        moved, vz, vy, vx = registration_imwarp_fields(reference, moving, alpha=args.alpha, levels=args.levels, verbose=args.verbose)

    return moved, vz, vy, vx


def process_blocks(fixed_image_np, moving_image_np, args, original_shape=None):
    # Original shape to be used for upsampling
    if original_shape is None:
        original_shape = fixed_image_np.shape

    fixed_blocks, block_size, factors = split_image(fixed_image_np, args.factors)
    moving_blocks, _, _ = split_image(moving_image_np, args.factors)
    
    print(f"Number of blocks: {len(fixed_blocks)}")
    print_memory_usage("After splitting")
    
    registered_blocks = []
    displacement_fields_vz = []
    displacement_fields_vy = []
    displacement_fields_vx = []
    
    for i, (fixed_block, moving_block) in enumerate(zip(fixed_blocks, moving_blocks)):
        print(f"Processing block {i + 1}/{len(fixed_blocks)} of size {fixed_block.shape}")
        # DEEDS correction
        moved, vz, vy, vx = calculates_deformation(fixed_block, moving_block, args, method='DEEDs')
        registered_blocks.append(moved)
        displacement_fields_vz.append(vz)
        displacement_fields_vy.append(vy)
        displacement_fields_vx.append(vx)
        
        print_memory_usage(f"After processing block {i + 1}")
    
    registered_image_np = stitch_blocks(registered_blocks, factors, block_size, fixed_image_np.shape)
    vz_stitched = stitch_blocks(displacement_fields_vz, factors, block_size, fixed_image_np.shape)
    vy_stitched = stitch_blocks(displacement_fields_vy, factors, block_size, fixed_image_np.shape)
    vx_stitched = stitch_blocks(displacement_fields_vx, factors, block_size, fixed_image_np.shape)
    
    displacement_fields_np = np.stack([vz_stitched, vy_stitched, vx_stitched], axis=-1)

    # Print shape before upscaling
    print(f"Displacement field shape before upsampling: {displacement_fields_np.shape}")
    print(f"Original shape: {original_shape}")
    
    # Upsample displacement field back to the original shape if binning was applied
    if args.binning_factor_xy > 1 or args.binning_factor_z > 1:
        print(f"$ Upsampling the deformation field to match the original image dimensions.")
        zoom_factors = [original_shape[0] / displacement_fields_np.shape[0],  # Z upsampling
                        original_shape[1] / displacement_fields_np.shape[1],  # Y upsampling
                        original_shape[2] / displacement_fields_np.shape[2]]  # X upsampling
        print(f"Zoom factors: {zoom_factors}")
        displacement_fields_np = zoom(displacement_fields_np, zoom_factors + [1], order=1)

    # Print shape after upscaling
    print(f"Displacement field shape after upsampling: {displacement_fields_np.shape}")

    return registered_image_np, displacement_fields_np

def smooth_vector_field(vector_field, sigma=1.0):
    """
    Smooth a 3D vector field using a Gaussian filter.

    Parameters:
    vector_field : numpy.ndarray
        The 3D vector field to be smoothed. Shape should be (z, y, x, 3) for 3D vectors.
    sigma : float
        The standard deviation for the Gaussian kernel. Higher values mean more smoothing.

    Returns:
    smoothed_field : numpy.ndarray
        The smoothed vector field.
    """
    smoothed_field = np.zeros_like(vector_field)
    
    # Apply Gaussian smoothing to each component of the vector field
    for i in range(vector_field.shape[-1]):
        smoothed_field[..., i] = gaussian_filter(vector_field[..., i], sigma=sigma)
    
    return smoothed_field


# Additional functions for plotting
def compute_intensity(displacement_field, z_plane):
    dz = displacement_field[z_plane, :, :, 0]
    dy = displacement_field[z_plane, :, :, 1]
    dx = displacement_field[z_plane, :, :, 2]
    intensity = np.sqrt(dx**2 + dy**2 + dz**2)
    return intensity,dx,dy,dz

def plot_deformation_intensity(displacement_field, z_plane, output_prefix):
    intensity,_,_,_ = compute_intensity(displacement_field, z_plane)
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity, cmap='Reds')
    plt.colorbar(label='Vector Field Intensity')
    plt.title(f'Intensity of Vector Field at Z-plane {z_plane}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f"{output_prefix}_intensity_z{z_plane}.png")
    plt.show()

def plot_deformation_intensity_xyz(displacement_field, z_plane, output_prefix):
    data = compute_intensity(displacement_field, z_plane)
    titles = ["dx^2+dx^2+dz^2", "dx", "dy", "dz"]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    ax = axes.ravel()

    for axis, img, title in zip(ax, data, titles):
        if "dx^2+dx^2+dz^2" not in title:
            min_max = np.max( [np.abs(np.max(img)), np.abs(np.min(img))] )
            vmin, vmax = -min_max, min_max
            cmap="bwr"
        else:
            vmin, vmax = 0, np.max(img)
            cmap="YlOrRd"
            
        im = axis.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)    
        axis.set_xlabel('X-axis')
        axis.set_ylabel('Y-axis')
        cbar1 = fig.colorbar(im, ax=axis, shrink=0.5)
        cbar1.set_label('pixels')

    fig.tight_layout()
    fig.suptitle(f'Intensity of Vector Field at Z-plane {z_plane}')

    fig.savefig(f"{output_prefix}_DF_intensity_z{z_plane}.png")

def compute_direction(displacement_field, z_plane):
    dz = displacement_field[z_plane, :, :, 0]
    dy = displacement_field[z_plane, :, :, 1]
    dx = displacement_field[z_plane, :, :, 2]
    direction = np.arctan2(dy, dx)
    norm = plt.Normalize(-np.pi, np.pi)
    direction_normalized = norm(direction)
    return direction_normalized

def plot_deformation_direction(displacement_field, z_plane, output_prefix):
    direction = compute_direction(displacement_field, z_plane)
    plt.figure(figsize=(10, 8))
    plt.imshow(direction, cmap='twilight', alpha=0.9, norm=plt.Normalize(-np.pi, np.pi))
    cbar = plt.colorbar(ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    cbar.set_label('Vector Field Direction (radians)')
    plt.title(f'Direction of Vector Field at Z-plane {z_plane}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f"{output_prefix}_DF_direction_z{z_plane}.png")
    plt.show()

#### apply DF ####
def sitk_warp(image_np, df_np):
    image = sitk.GetImageFromArray(image_np.astype(np.float32))
    field = sitk.GetImageFromArray(df_np.astype(np.float32), isVector=True)
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(sitk.sitkLinear)

    warped = warp_filter.Execute(image, field)

    return sitk.GetArrayFromImage(warped)
    
#def applyDF(new_image_np, displacement_fields_np, reference_shape):
    #  resample new image to match DF grid
    #new_image_np = zoom(  new_image_np,(
            #reference_shape[0] / new_image_np.shape[0],
            #reference_shape[1] / new_image_np.shape[1],
            #reference_shape[2] / new_image_np.shape[2],),order=1)
    #  apply deformation field
    #warped_new_np = sitk_warp(new_image_np, displacement_fields_np)
    #return warped_new_np
    
def apply_displacement_field(image, displacement_field):
    """
    Apply the displacement field to the input image.
    
    Parameters:
    image (SimpleITK.Image): The input image.
    displacement_field (SimpleITK.Image): The displacement field.

    Returns:
    SimpleITK.Image: The deformed image.
    """
    # Cast the displacement field to the required type
    displacement_field = sitk.Cast(displacement_field, sitk.sitkVectorFloat64)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.DisplacementFieldTransform(displacement_field))
    
    deformed_image = resampler.Execute(image)
    return deformed_image
    
class BothImgRbgFile:
    def __init__(self, image1, image2, tag='', title=''):
        self.image1 = image1
        self.image2 = image2
        self.tag = tag
        if title is None:
            self.title = tag  # gets title from tag
        else:
            self.title = title  # New attribute to hold the title

    def save(self, folder_path, basename):
        self.folder_path = folder_path
        self.basename = f"{basename}_{self.tag}_overlay"
        self.path_name = os.path.join(self.folder_path, self.basename + ".png")
        
        # Normalize images and rescale intensity
        img_1 = self.image1 / self.image1.max()
        img_2 = self.image2 / self.image2.max()
        img_1 = exposure.rescale_intensity(img_1, out_range=(0, 1))
        img_2 = exposure.rescale_intensity(img_2, out_range=(0, 1))
        
        # Create the figure and axis
        fig, ax1 = plt.subplots()
        fig.set_size_inches((30, 30))
        
        # Create RGB overlay image
        null_image = np.zeros(img_1.shape)
        rgb = np.dstack([img_1, img_2, null_image])
        
        # Display the image and set the title
        ax1.imshow(rgb)
        ax1.axis("off")
        ax1.set_title(self.title)  # Set the title of the figure
        
        # Save the figure
        fig.savefig(self.path_name)
        plt.close(fig)


def plot_4_images(allimages, titles=None):
    if titles is None:
        titles = [
            "reference",
            "cycle <i>",
            "processed reference",
            "processed cycle <i>",
        ]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    ax = axes.ravel()

    for axis, img, title in zip(ax, allimages, titles):
        axis.imshow(img, cmap="Greys")
        axis.set_title(title)
    fig.tight_layout()

    return fig

def plots_normalized_images(image_ref, image_ref_0, image_3d_0, image_3d, path_name_normalized):

    images_raw = [image_ref_0, image_3d_0]
    images_processed = [image_ref, image_3d]

    images_raw_2d = [np.sum(x, axis=0) for x in images_raw]
    images_processed_2d = [np.sum(x, axis=0) for x in images_processed]    
    fig1 = plot_4_images(
        images_raw_2d + images_processed_2d,
        titles=["reference", "cycle <i>", "processed reference", "processed cycle <i>"],
    )

    print(f"$ saved: {path_name_normalized}")    
    fig1.savefig(path_name_normalized)



def main():
    parser = argparse.ArgumentParser(description="Align two 3D images using DEEDS with block-wise processing.")
    parser.add_argument('--reference', required=True, help='Path to the reference (fixed) image file.')
    parser.add_argument('--moving', required=True, help='Path to the moving image file.')
    parser.add_argument('--tomove', required=True, help='Path to the image NUMPY format that needs to be moved with DF.')
    parser.add_argument('--output', required=True, help='Path to the output (aligned) image file.')
    parser.add_argument('--png_folder', default='', help='Path to the folder that will hold png output files.')
    parser.add_argument('--displacement_field', required=True, help='Path to save the displacement field image file.')
    parser.add_argument('--displacement_format', choices=['tif', 'nii', 'h5'], default='h5', help='Format to save the displacement field. Default is h5.')
    parser.add_argument('--factors', type=int, nargs=2, default=[2, 2], help='Factors to split the image (y, x). Default: 2 2 (two blocks by two blocks)')
    parser.add_argument('--alpha', type=float, default=1.6, help='alpha factor. Default=1.6')
    parser.add_argument('--levels', type=int, default=5, help='number of levels. Default=5')
    parser.add_argument('--shifts', type=float, nargs=3, default=[0, 0, 0], help='Shifts to apply to the moving image in XYZ plane before registration. Default: [0, 0, 0]')
    parser.add_argument('--z_binning', type=int, default=2, help='Reinterpolates the image by keeping only one every z_binning planes. Default is 2.')
    parser.add_argument('--lower_threshold', type=float, default=0.9, help='Lower threshold for intensity adjustment in preprocessing.')
    parser.add_argument('--higher_threshold', type=float, default=0.9999999, help='Higher threshold for intensity adjustment in preprocessing.')
    parser.add_argument('--binning_factor_xy', type=int, default=1, help='Factor to downsample the XY dimensions before processing. Default is 1 (no binning).')
    parser.add_argument('--binning_factor_z', type=int, default=1, help='Factor to downsample the Z dimension before processing. Default is 1 (no binning).')
    parser.add_argument("--verbose", help="Default=False", action='store_true')
    parser.add_argument('--smooth_DF', type=float, default=3.0, help='Smooth deformation field radius. Default = 3 px. Use 0.0 to avoid smoothing.')

    args = parser.parse_args()

    print("This algorithm aligns two 3D volumes using the DEEDS method (deformable) based on optical flow.")
    print("This implementation breaks the images in user-defined blocks.")    
    print(f"$ Will save pngs at: {args.png_folder}")

    # loads images into memory
    check_file_existence(args.reference, args.moving)
    
    fixed_image = read_image(args.reference)
    moving_image = read_image(args.moving)

    fixed_image_np_0 = to_numpy(fixed_image)
    moving_image_np_0 = to_numpy(moving_image)
    
    original_shape = fixed_image_np_0.shape

    # pre-processes images
    fixed_image_np, moving_image_np = preprocess_images(fixed_image_np_0, moving_image_np_0, args)
    fixed_image = to_sitk(fixed_image_np)

    # Plot the normalized images
    png_path = os.path.join(args.png_folder, os.path.basename(args.output))
    plots_normalized_images(fixed_image_np_0, fixed_image_np, moving_image_np_0, moving_image_np, png_path.split('.')[0] + "_normalized.png")

    # registers images
    registered_image_np, displacement_fields_np = process_blocks(fixed_image_np, moving_image_np, args)

    # Check dimensions after upsampling
    print(f"$ -->size DF, registered_image: {displacement_fields_np.shape, registered_image_np}")

    # Ensure we create the displacement field with isVector=True to handle the vector field properly
    displacement_field_sitk = sitk.GetImageFromArray(displacement_fields_np, isVector=True)
    registered_image_sitk = to_sitk(registered_image_np, ref_img=fixed_image)
    
    # Copy information from the reference image to the displacement field
    try:
        displacement_field_sitk.CopyInformation(fixed_image)  # Copy spacing, origin, and direction
    except RuntimeError as e:
        print(f"Error during CopyInformation: {e}")
        print(f"Displacement field size: {displacement_field_sitk.GetSize()}\nFixed image size: {fixed_image.GetSize()}")
        
    # writes displacement field
    write_displacement_field(displacement_field_sitk, args.displacement_field, args.displacement_format)
    print(f"Displacement field written to {args.displacement_field}")

    # Plot the overlay of the reference and registered images
    overlay = BothImgRbgFile(fixed_image_np.max(axis=0), moving_image_np.max(axis=0), tag='reference_original')
    overlay.save(args.png_folder, os.path.basename(args.output))

    overlay = BothImgRbgFile(fixed_image_np.max(axis=0), registered_image_np.max(axis=0), tag='reference_aligned')
    overlay.save(args.png_folder, os.path.basename(args.output))

    # Plot the intensity and direction of the deformation field at the center z-plane
    z_plane = registered_image_np.shape[0] // 2
    plot_deformation_intensity_xyz(displacement_fields_np, z_plane, png_path)
    plot_deformation_direction(displacement_fields_np, z_plane, png_path)

    if args.smooth_DF>0.0:
        plot_deformation_intensity_xyz(smoothed_displacement_fields_np, z_plane, png_path.split('.')[0] + "_smoothed")

    # apply DF to mask ch01 as numpy
    registered_image = applyDF(
        args.tomove,
        displacement_fields_np,
        registered_image.shape)

    warped_new_sitk = to_sitk(warped_new_np, ref_img=fixed_image)
    write_image(warped_new_sitk, "name_image_corrected.tif")

if __name__ == "__main__":
    main()
