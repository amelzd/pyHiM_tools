#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.signal import correlate, correlation_lags


def read_image(path: str) -> Tuple[sitk.Image, np.ndarray]:
    """Read a TIFF image with SimpleITK and return both the image and a NumPy array (z, y, x)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find image at {path}")
    image = sitk.ReadImage(path)
    image_np = sitk.GetArrayFromImage(image)
    return image, image_np


def laplacian_variance_profile(image_np: np.ndarray) -> np.ndarray:
    """Compute the Laplacian variance profile for a 3D stack (z, y, x)."""

    """ 
    variances = []
    for plane in image_np:
        lap = ndimage.laplace(plane.astype(np.float32))
        variances.append(np.var(lap))
    """
    from skimage import filters

    raw_images = [image_np[i, :, :] for i in range(image_np.shape[0])]
    laplacian_variance = [np.var(filters.laplace(img)) for img in raw_images]
    laplacian_variance = laplacian_variance / max(laplacian_variance)
    return np.asarray(laplacian_variance)


def estimate_imaging_range(profile: np.ndarray, threshold_ratio: float = 0.1) -> Tuple[int, int]:
    """Estimate imaging range (start, end indices) based on a threshold of the profile's maximum."""
    if profile.size == 0:
        return (0, 0)
    peak = float(np.max(profile))
    if peak == 0.0:
        return (0, profile.size - 1)
    threshold = peak * threshold_ratio
    valid = np.where(profile >= threshold)[0]
    if valid.size == 0:
        return (0, profile.size - 1)
    return int(valid[0]), int(valid[-1])


def propose_z_shift(ref_range: Tuple[int, int], target_range: Tuple[int, int]) -> int:
    """Propose an integer z-shift to align target to reference using range centers."""
    ref_center = 0.5 * (ref_range[0] + ref_range[1])
    target_center = 0.5 * (target_range[0] + target_range[1])
    return int(round(ref_center - target_center))


def cross_correlation_z_shift(ref_profile: np.ndarray, target_profile: np.ndarray) -> int:
    """Estimate z-shift by maximizing cross-correlation between two 1D profiles.

    A positive shift means the target stack needs to be moved towards higher z indices
    (rolled forward) to best align with the reference profile.
    """

    if ref_profile.size == 0 or target_profile.size == 0:
        return 0

    ref_centered = ref_profile - np.mean(ref_profile)
    target_centered = target_profile - np.mean(target_profile)

    corr = correlate(ref_centered, target_centered, mode="full")
    lags = correlation_lags(ref_centered.size, target_centered.size, mode="full")
    best_lag = int(lags[int(np.argmax(corr))])
    return best_lag


def apply_z_shift(image_np: np.ndarray, shift: int) -> np.ndarray:
    """Shift the z-axis by the requested amount using roll (positive moves target towards higher z)."""
    if shift == 0:
        return image_np
    shifted = np.roll(image_np, shift=shift, axis=0)
    if shift > 0:
        shifted[:shift] = 0
    else:
        shifted[shift:] = 0
    return shifted


def plot_profiles(
    profiles: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    output_path: str,
    ranges: Sequence[Tuple[int, int]] | None = None,
) -> None:
    """Plot one or more z-profiles and save to disk."""
    plt.figure(figsize=(8, 4))
    for i, profile in enumerate(profiles):
        plt.plot(profile, label=labels[i])
        if ranges is not None and i < len(ranges):
            start, end = ranges[i]
            plt.axvspan(start, end, alpha=0.15)
    plt.xlabel("Z index")
    plt.ylabel("Laplacian variance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def estimate_z_shift_cross_correlation(ref_image: sitk.Image, target_image: sitk.Image) -> float:
    """Estimate z-shift using SimpleITK correlation metric in a translation-only registration."""
    # Cast images to a floating type because the correlation metric does not
    # support unsigned 16-bit inputs.
    ref_image = sitk.Cast(ref_image, sitk.sitkFloat32)
    target_image = sitk.Cast(target_image, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    transform = sitk.TranslationTransform(ref_image.GetDimension())
    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScales([1.0, 1.0, 1.0])
    final_transform = registration.Execute(ref_image, target_image)
    return final_transform.GetOffset()[2]


def summarize_range(range_tuple: Tuple[int, int]) -> str:
    return f"[{range_tuple[0]}, {range_tuple[1]}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate z-shift between two channels using imaging range and cross-correlation.")
    parser.add_argument("--reference", required=True, help="Reference channel TIFF image.")
    parser.add_argument("--target", required=True, help="Target channel TIFF image to be shifted.")
    parser.add_argument("--output", required=True, help="Output directory for plots and logs.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold ratio (0-1) for imaging range detection.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Reading reference: {args.reference}")
    ref_image, ref_np = read_image(args.reference)
    print(f"Reading target: {args.target}")
    target_image, target_np = read_image(args.target)

    ref_profile = laplacian_variance_profile(ref_np)
    target_profile = laplacian_variance_profile(target_np)

    ref_range = estimate_imaging_range(ref_profile, args.threshold)
    target_range = estimate_imaging_range(target_profile, args.threshold)

    range_plot = os.path.join(args.output, "z_profile_initial.png")
    plot_profiles(
        [ref_profile, target_profile],
        [f"Reference range {summarize_range(ref_range)}", f"Target range {summarize_range(target_range)}"],
        "Imaging range (initial)",
        range_plot,
        ranges=[ref_range, target_range],
    )
    print(f"Saved initial range plot to {range_plot}")

    proposed_shift = cross_correlation_z_shift(ref_profile, target_profile)
    shifted_target_np = apply_z_shift(target_np, proposed_shift)
    shifted_profile = laplacian_variance_profile(shifted_target_np)
    shifted_range = estimate_imaging_range(shifted_profile, args.threshold)

    adjusted_plot = os.path.join(args.output, "z_profile_shifted.png")
    plot_profiles(
        [ref_profile, shifted_profile],
        [f"Reference range {summarize_range(ref_range)}", f"Shifted target range {summarize_range(shifted_range)}"],
        f"Imaging range after proposed shift (z shift = {proposed_shift})",
        adjusted_plot,
        ranges=[ref_range, shifted_range],
    )
    print(f"Proposed z-shift based on profile cross-correlation: {proposed_shift}")
    print(f"Saved shifted range plot to {adjusted_plot}")

    cc_shift = estimate_z_shift_cross_correlation(ref_image, target_image)
    print(f"Estimated z-shift from global cross-correlation: {cc_shift:.3f}")


if __name__ == "__main__":
    main()
