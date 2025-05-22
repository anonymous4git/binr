import os
import argparse
import tempfile
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Nipype MRtrix3 interfaces
from nipype.interfaces.mrtrix3.utils import MRConvert
from nipype.interfaces.mrtrix3 import BrainMask, TensorMetrics
from nipype.interfaces.mrtrix3.reconst import FitTensor


def run_mrtrix_dti(dwi_nii: str, bvals_path: str, bvecs_path: str, out_dir: str):
    """Run MRtrix3's DTI workflow (dwi2tensor + tensor2metric) via Nipype.

    Parameters
    ----------
    dwi_nii : str
        Path to the 4‑D diffusion NIfTI image.
    bvals_path : str
        Path to the *bvals* file.
    bvecs_path : str
        Path to the *bvecs* file.
    out_dir : str
        Working/output directory.

    Returns
    -------
    md_path, fa_path : str
        Paths to the generated MD and FA NIfTI maps.
    """
    # 1) Convert input NIfTI to MRtrix .mif (faster I/O)
    dwi_mif = os.path.join(out_dir, "dwi.mif")
    MRConvert(
        in_file=dwi_nii,
        out_file=dwi_mif,
        grad_fsl=(bvecs_path, bvals_path)  # Add gradient information
    ).run()

    # 2) Compute a brain mask
    mask_mif = os.path.join(out_dir, "mask.mif")
    BrainMask(in_file=dwi_mif, out_file=mask_mif).run()

    # 3) Fit diffusion tensor
    tensor_mif = os.path.join(out_dir, "dti.mif")
    FitTensor(
        in_file=dwi_mif,
        out_file=tensor_mif,
        in_bval=bvals_path,
        in_bvec=bvecs_path,
        in_mask=mask_mif,
    ).run()

    # 4) Derive MD & FA maps
    fa_mif = os.path.join(out_dir, "fa.mif")
    md_mif = os.path.join(out_dir, "md.mif")
    TensorMetrics(
        in_file=tensor_mif,
        out_fa=fa_mif,
        out_adc=md_mif,
        in_mask=mask_mif  # Ensure metrics are only computed within the brain
    ).run()

    # 5) Convert maps back to NIfTI‑GZ for downstream use
    fa_nii = os.path.join(out_dir, "fa.nii.gz")
    md_nii = os.path.join(out_dir, "md.nii.gz")
    MRConvert(in_file=fa_mif, out_file=fa_nii).run()
    MRConvert(in_file=md_mif, out_file=md_nii).run()

    return md_nii, fa_nii


def compute_md_fa(dwi_img: nib.Nifti1Image, bvals_path: str, bvecs_path: str):
    """Compute MD & FA for the given DWI using MRtrix3.

    The heavy lifting is delegated to `run_mrtrix_dti`, which relies on
    *dwi2tensor* and *tensor2metric*. We work in a temporary directory and
    return NumPy arrays ready for evaluation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dwi_path = os.path.join(tmpdir, "dwi.nii.gz")
        nib.save(dwi_img, dwi_path)

        md_path, fa_path = run_mrtrix_dti(dwi_path, bvals_path, bvecs_path, tmpdir)

        md = nib.load(md_path).get_fdata().astype(np.float32)
        fa = nib.load(fa_path).get_fdata().astype(np.float32)

    return md, fa


def compute_diff_metrics(pred_md, pred_fa, gt_md, gt_fa, mask):
    """Compute difference metrics between predicted and GT MD/FA maps.

    Metrics are computed only within the supplied mask.
    Returns absolute difference maps (unmasked) for optional saving.
    """
    # Apply mask to all arrays
    pred_md_masked = pred_md[mask]
    pred_fa_masked = pred_fa[mask]
    gt_md_masked = gt_md[mask]
    gt_fa_masked = gt_fa[mask]

    # Absolute differences
    diff_md = np.abs(pred_md - gt_md)
    diff_fa = np.abs(pred_fa - gt_fa)

    # Classical error metrics within mask
    mse_md = np.mean(diff_md[mask] ** 2)
    mse_fa = np.mean(diff_fa[mask] ** 2)
    mae_md = np.mean(diff_md[mask])
    mae_fa = np.mean(diff_fa[mask])

    # Relative (%) errors within mask
    rel_md = diff_md[mask] / (gt_md_masked + 1e-6) * 100.0
    rel_fa = diff_fa[mask] / (gt_fa_masked + 1e-6) * 100.0

    # PSNR & SSIM within mask
    psnr_md = psnr(gt_md_masked, pred_md_masked, data_range=gt_md_masked.max() - gt_md_masked.min())
    psnr_fa = psnr(gt_fa_masked, pred_fa_masked, data_range=gt_fa_masked.max() - gt_fa_masked.min())
    ssim_md = ssim(gt_md_masked, pred_md_masked, data_range=gt_md_masked.max() - gt_md_masked.min())
    ssim_fa = ssim(gt_fa_masked, pred_fa_masked, data_range=gt_fa_masked.max() - gt_fa_masked.min())

    return (diff_md, diff_fa,
            mse_md, mse_fa,
            mae_md, mae_fa,
            rel_md, rel_fa,
            psnr_md, psnr_fa,
            ssim_md, ssim_fa)


def print_distribution(arr, name):
    """Utility to print a 5‑number summary and 95% IQR of a (masked) 1‑D array."""
    arr_sorted = np.sort(arr)
    print(f"{name} Relative Error Distribution:")
    print(f"Min: {arr_sorted[0]:.4f}")
    print(f"25th percentile: {np.percentile(arr_sorted, 25):.4f}")
    print(f"Median: {np.percentile(arr_sorted, 50):.4f}")
    print(f"75th percentile: {np.percentile(arr_sorted, 75):.4f}")
    print(f"Max: {arr_sorted[-1]:.4f}")
    # Calculate and print 95% IQR
    iqr_95_low = np.percentile(arr_sorted, 2.5)
    iqr_95_high = np.percentile(arr_sorted, 97.5)
    print(f"95% IQR: [{iqr_95_low:.4f}, {iqr_95_high:.4f}]")


def save_metrics_to_csv(metrics_dict, output_dir):
    """Save MD/FA metrics to CSV files.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing metrics for MD and FA
    output_dir : str
        Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for whole-brain metrics
    whole_brain_metrics = {
        'metric': ['MD', 'FA'],
        'mse': [metrics_dict['mse_md'], metrics_dict['mse_fa']],
        'mae': [metrics_dict['mae_md'], metrics_dict['mae_fa']],
        'mean_rel_err': [np.mean(metrics_dict['rel_md']), np.mean(metrics_dict['rel_fa'])],
        'psnr': [metrics_dict['psnr_md'], metrics_dict['psnr_fa']],
        'ssim': [metrics_dict['ssim_md'], metrics_dict['ssim_fa']],
        'iqr_95_low': [np.percentile(metrics_dict['rel_md'], 2.5), np.percentile(metrics_dict['rel_fa'], 2.5)],
        'iqr_95_high': [np.percentile(metrics_dict['rel_md'], 97.5), np.percentile(metrics_dict['rel_fa'], 97.5)]
    }
    
    # Create DataFrame for white matter metrics if available
    if 'mse_fa_wm' in metrics_dict:
        wm_metrics = {
            'metric': ['FA (WM)'],
            'mse': [metrics_dict['mse_fa_wm']],
            'mae': [metrics_dict['mae_fa_wm']],
            'mean_rel_err': [np.mean(metrics_dict['rel_fa_wm'])],
            'psnr': [metrics_dict['psnr_fa_wm']],
            'ssim': [metrics_dict['ssim_fa_wm']],
            'iqr_95_low': [np.percentile(metrics_dict['rel_fa_wm'], 2.5)],
            'iqr_95_high': [np.percentile(metrics_dict['rel_fa_wm'], 97.5)]
        }
    
    # Save to CSV
    pd.DataFrame(whole_brain_metrics).to_csv(os.path.join(output_dir, 'md_fa_metrics_whole_brain.csv'), index=False)
    if 'mse_fa_wm' in metrics_dict:
        pd.DataFrame(wm_metrics).to_csv(os.path.join(output_dir, 'md_fa_metrics_wm.csv'), index=False)


def main(args):
    dwi_dir = os.path.join(args.results_dir, "dwi")

    pred_dwi_path = os.path.join(dwi_dir, "dwi_pred.nii.gz")
    gt_dwi_path = os.path.join(dwi_dir, "dwi_gt.nii.gz")
    bvals_path = os.path.join(dwi_dir, "bvals.txt")
    bvecs_path = os.path.join(dwi_dir, "bvecs.txt")

    # ----- Prediction -----
    if not args.load_maps:
        pred_md, pred_fa = compute_md_fa(nib.load(pred_dwi_path), bvals_path, bvecs_path)
    else:
        pred_md = nib.load(os.path.join(dwi_dir, "md_pred.nii.gz")).get_fdata().astype(np.float32)
        pred_fa = nib.load(os.path.join(dwi_dir, "fa_pred.nii.gz")).get_fdata().astype(np.float32)

    # ----- Ground‑truth -----
    if not args.load_maps:
        gt_md, gt_fa = compute_md_fa(nib.load(gt_dwi_path), bvals_path, bvecs_path)
    else:
        gt_md = nib.load(os.path.join(dwi_dir, "md_gt.nii.gz")).get_fdata().astype(np.float32)
        gt_fa = nib.load(os.path.join(dwi_dir, "fa_gt.nii.gz")).get_fdata().astype(np.float32)

    # ----- Brain Mask -----
    if not args.load_maps:
        with tempfile.TemporaryDirectory() as tmpdir:
            dwi_mif = os.path.join(tmpdir, "dwi.mif")
            MRConvert(
                in_file=pred_dwi_path,
                out_file=dwi_mif,
                grad_fsl=(bvecs_path, bvals_path)
            ).run()
            mask_mif = os.path.join(tmpdir, "mask.mif")
            mask_nii = os.path.join(tmpdir, "mask.nii.gz")
            BrainMask(in_file=dwi_mif, out_file=mask_mif).run()
            MRConvert(in_file=mask_mif, out_file=mask_nii).run()
            mask = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_mif = os.path.join(dwi_dir, "mask.mif")
            mask_nii = os.path.join(tmpdir, "mask.nii.gz")
            MRConvert(in_file=mask_mif, out_file=mask_nii).run()
            mask = nib.load(mask_nii).get_fdata().astype(bool)

    # ----- Metrics (whole brain) -----
    (diff_md, diff_fa,
     mse_md, mse_fa,
     mae_md, mae_fa,
     rel_md, rel_fa,
     psnr_md, psnr_fa,
     ssim_md, ssim_fa) = compute_diff_metrics(pred_md, pred_fa, gt_md, gt_fa, mask)

    # Store metrics in a dictionary
    metrics_dict = {
        'mse_md': mse_md, 'mse_fa': mse_fa,
        'mae_md': mae_md, 'mae_fa': mae_fa,
        'rel_md': rel_md, 'rel_fa': rel_fa,
        'psnr_md': psnr_md, 'psnr_fa': psnr_fa,
        'ssim_md': ssim_md, 'ssim_fa': ssim_fa
    }

    print("=== Diffusion Metric Evaluation (whole‑brain) ===")
    print(f"MD | MSE: {mse_md:.6f} | MAE: {mae_md:.6f} | Mean Rel Err: {np.mean(rel_md):.6f} | PSNR: {psnr_md:.2f} dB | SSIM: {ssim_md:.4f}")
    print(f"FA | MSE: {mse_fa:.6f} | MAE: {mae_fa:.6f} | Mean Rel Err: {np.mean(rel_fa):.6f} | PSNR: {psnr_fa:.2f} dB | SSIM: {ssim_fa:.4f}\n")

    # ----- White‑matter mask (GT FA threshold) -----
    wm_mask = (gt_fa > args.fa_thresh) & mask
    n_wm = int(np.sum(wm_mask))
    if n_wm == 0:
        print(f"Warning: WM mask is empty with threshold {args.fa_thresh:.2f}. Skipping WM evaluation.\n")
    else:
        diff_fa_wm = np.abs(pred_fa - gt_fa)
        mse_fa_wm = np.mean(diff_fa_wm[wm_mask] ** 2)
        mae_fa_wm = np.mean(diff_fa_wm[wm_mask])
        rel_fa_wm = diff_fa_wm[wm_mask] / (gt_fa[wm_mask] + 1e-6) * 100.0
        psnr_fa_wm = psnr(gt_fa[wm_mask], pred_fa[wm_mask], data_range=gt_fa[wm_mask].max() - gt_fa[wm_mask].min())
        ssim_fa_wm = ssim(gt_fa[wm_mask], pred_fa[wm_mask], data_range=gt_fa[wm_mask].max() - gt_fa[wm_mask].min())

        # Add WM metrics to dictionary
        metrics_dict.update({
            'mse_fa_wm': mse_fa_wm,
            'mae_fa_wm': mae_fa_wm,
            'rel_fa_wm': rel_fa_wm,
            'psnr_fa_wm': psnr_fa_wm,
            'ssim_fa_wm': ssim_fa_wm
        })

        print("=== FA Evaluation in White Matter (FA > {:.2f}) ===".format(args.fa_thresh))
        print(f"Voxels: {n_wm}")
        print(f"FA | MSE: {mse_fa_wm:.6f} | MAE: {mae_fa_wm:.6f} | Mean Rel Err: {np.mean(rel_fa_wm):.6f} | PSNR: {psnr_fa_wm:.2f} dB | SSIM: {ssim_fa_wm:.4f}\n")
        print_distribution(rel_fa_wm, "FA (WM)")

    # ----- Diagnostic information -----
    print("\n=== MD Diagnostic Information ===")
    print(f"GT MD - Min: {gt_md[mask].min():.4f}, Max: {gt_md[mask].max():.4f}, Mean: {gt_md[mask].mean():.4f}")
    print(f"Pred MD - Min: {pred_md[mask].min():.4f}, Max: {pred_md[mask].max():.4f}, Mean: {pred_md[mask].mean():.4f}")

    print("\n=== FA Diagnostic Information ===")
    print(f"GT FA - Min: {gt_fa[mask].min():.4f}, Max: {gt_fa[mask].max():.4f}, Mean: {gt_fa[mask].mean():.4f}")
    print(f"Pred FA - Min: {pred_fa[mask].min():.4f}, Max: {pred_fa[mask].max():.4f}, Mean: {pred_fa[mask].mean():.4f}")

    # ----- Relative error distribution -----
    print_distribution(rel_md, "MD")
    print()
    print_distribution(rel_fa, "FA")

    # ----- Small GT value warnings -----
    for gt_vals, pred_vals, rel_err, name in [
        (gt_md[mask], pred_md[mask], rel_md, "MD"),
        (gt_fa[mask], pred_fa[mask], rel_fa, "FA")
    ]:
        small_gt = gt_vals < 0.01
        if np.any(small_gt):
            print(f"\nWarning: Found {np.sum(small_gt)} voxels with very small GT {name} values (< 0.01)")
            print(f"These voxels have mean relative error: {np.mean(rel_err[small_gt]):.4f}")
            print(f"Mean GT {name} in these voxels: {np.mean(gt_vals[small_gt]):.4f}")
            print(f"Mean Pred {name} in these voxels: {np.mean(pred_vals[small_gt]):.4f}")

    # ----- Save outputs -----
    dwi_affine = nib.load(gt_dwi_path).affine
    nib.save(nib.Nifti1Image(diff_md, dwi_affine), os.path.join(args.results_dir, "diff_md.nii.gz"))
    nib.save(nib.Nifti1Image(diff_fa, dwi_affine), os.path.join(args.results_dir, "diff_fa.nii.gz"))

    if not args.load_maps:
        nib.save(nib.Nifti1Image(pred_md, dwi_affine), os.path.join(dwi_dir, "md_pred.nii.gz"))
        nib.save(nib.Nifti1Image(pred_fa, dwi_affine), os.path.join(dwi_dir, "fa_pred.nii.gz"))
        nib.save(nib.Nifti1Image(gt_md, dwi_affine), os.path.join(dwi_dir, "md_gt.nii.gz"))
        nib.save(nib.Nifti1Image(gt_fa, dwi_affine), os.path.join(dwi_dir, "fa_gt.nii.gz"))
        # Save masks for convenience
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), dwi_affine), os.path.join(dwi_dir, "mask_brain.nii.gz"))
        nib.save(nib.Nifti1Image(wm_mask.astype(np.uint8), dwi_affine), os.path.join(dwi_dir, "mask_wm.nii.gz"))

    # Save metrics to CSV
    save_metrics_to_csv(metrics_dict, args.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MD/FA via MRtrix3 and evaluate prediction error (whole‑brain & WM)")
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory with a dwi/ sub‑folder.")
    parser.add_argument("--load_maps", action="store_true", help="Load precomputed MD/FA maps instead of computing them.")
    parser.add_argument("--fa_thresh", type=float, default=0.25, help="FA threshold for defining white matter (default: 0.25)")
    args = parser.parse_args()

    # Tell MRtrix to use all available threads
    os.environ.setdefault("MRTRIX_NTHREADS", "0")

    main(args)
