import os
import numpy as np
import nibabel as nib
import tempfile
import shutil
import matplotlib.pyplot as plt
import argparse
from nipype.interfaces import mrtrix3 as mrt

def setup_directories(save_folder, root_dir):
    """Set up the working directories and verify the existence of required input files.

    This function creates the output directory if it doesn't exist and checks for the
    presence of all required input files (DWI data, bvecs, bvals, and brain mask).

    Parameters
    ----------
    save_folder : str
        Name of the folder where results will be saved
    root_dir : str
        Path to the directory containing input DWI data files

    Returns
    -------
    tuple
        - here (str): Absolute path to the created output directory
        - dwi_nii (str): Path to the DWI data file
        - bvecs_txt (str): Path to the bvecs file
        - bvals_txt (str): Path to the bvals file
        - mask_nii (str): Path to the brain mask file

    Raises
    ------
    FileNotFoundError
        If any of the required input files are missing
    """
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_folder)
    if not os.path.exists(here):
        os.makedirs(here)
    
    dwi_nii = os.path.join(here, root_dir, 'data.nii.gz')
    bvecs_txt = os.path.join(here, root_dir, 'bvecs.txt')
    bvals_txt = os.path.join(here, root_dir, 'bvals.txt')
    mask_nii = os.path.join(here, root_dir, 'brain_mask.nii.gz')

    for f in (dwi_nii, bvecs_txt, bvals_txt, mask_nii):
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    
    print('All inputs found ✓')
    return here, dwi_nii, bvecs_txt, bvals_txt, mask_nii

def filter_bvalue(dwi_nii, bvals_txt, bvecs_txt, target_bval):
    """Filter DWI data to include both b0 volumes and volumes with a specific b-value.

    This function loads the DWI data and its corresponding bvals/bvecs files,
    then filters them to keep:
    1. b0 volumes (b-value ≈ 0)
    2. volumes with b-values within a tolerance of ±50 of the target b-value

    Parameters
    ----------
    dwi_nii : str
        Path to the DWI data file in NIfTI format
    bvals_txt : str
        Path to the b-values file
    bvecs_txt : str
        Path to the b-vectors file
    target_bval : int
        Target b-value to filter for (e.g., 1000, 2000, 3000)

    Returns
    -------
    tuple
        - filtered_dwi (ndarray): 4D array of filtered DWI data
        - filtered_bvals (ndarray): Array of filtered b-values
        - filtered_bvecs (ndarray): Array of filtered gradient directions
        - nii.affine (ndarray): Affine transformation matrix from the input NIfTI
        - bval_mask (ndarray): Boolean mask used for filtering

    Notes
    -----
    The function uses a tolerance of ±50 around the target b-value and ±10 around
    b0 to account for slight variations in the actual b-values.
    """
    bvals = np.loadtxt(bvals_txt)
    bvecs = np.loadtxt(bvecs_txt)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    # Create masks for b0 and target b-value with tolerances
    b0_mask = np.abs(bvals) < 10  # b0 images typically have small non-zero values
    target_mask = np.abs(bvals - target_bval) < 50
    bval_mask = b0_mask | target_mask  # Combine masks with OR operation
    
    filtered_bvals = bvals[bval_mask]
    filtered_bvecs = bvecs[:, bval_mask]

    # Load and filter DWI data
    nii = nib.load(dwi_nii)
    dwi_data = nii.get_fdata()
    filtered_dwi = dwi_data[..., bval_mask]
    
    print(f"Original volumes: {len(bvals)}")
    print(f"Filtered volumes: {len(filtered_bvals)} (b0: {np.sum(b0_mask)}, b={target_bval}: {np.sum(target_mask)})")
    return filtered_dwi, filtered_bvals, filtered_bvecs, nii.affine, bval_mask

def subsample_gradients(filtered_dwi, filtered_bvals, filtered_bvecs, here, nii_affine):
    """Subsample gradient directions by randomly selecting half of the directions for each b-value.

    This function performs stratified random sampling of the gradient directions,
    ensuring that we maintain good angular coverage while reducing the number of
    directions. It also saves the subsampled data and performs angular coverage analysis.

    Parameters
    ----------
    filtered_dwi : ndarray
        4D array of filtered DWI data
    filtered_bvals : ndarray
        Array of filtered b-values
    filtered_bvecs : ndarray
        Array of filtered gradient directions
    here : str
        Path to the output directory
    nii_affine : ndarray
        Affine transformation matrix for the NIfTI output

    Notes
    -----
    The function saves several files:
    - dwi_sub.nii.gz: Subsampled DWI data
    - sub_bvecs.txt: Subsampled gradient directions
    - sub_bvals.txt: Subsampled b-values
    - grad.txt: Combined gradient file in MRtrix format
    - seen_directions.npy: Indices of selected directions
    """
    sub_nii = os.path.join(here, 'dwi_sub.nii.gz')
    sub_bvecs_txt = os.path.join(here, 'sub_bvecs.txt')
    sub_bvals_txt = os.path.join(here, 'sub_bvals.txt')

    if os.path.exists(sub_bvecs_txt) and os.path.exists(sub_bvals_txt) and os.path.exists(sub_nii):
        print('Subsampled files already exist ✓')
        return

    np.random.seed(42)
    sub_idx = []
    for b in np.unique(filtered_bvals):
        idx_here = np.where(filtered_bvals == b)[0]
        # For b0 images, keep all of them. For other b-values, keep half
        if b < 10:  # b0 images
            keep = idx_here
        else:
            keep = np.random.choice(idx_here, size=max(1, len(idx_here)//2), replace=False)
        sub_idx.extend(keep)
    sub_idx = sorted(sub_idx)

    print(f'Subsampled {len(filtered_bvals)} → {len(sub_idx)} directions')
    
    # Angular coverage analysis - only for non-b0 directions
    non_b0_idx = [i for i in sub_idx if filtered_bvals[i] >= 10]
    if len(non_b0_idx) > 0:
        # Get the dot product between all pairs of non-b0 directions
        bvecs_non_b0 = filtered_bvecs[:, non_b0_idx]
        seen_angles = np.arccos(np.clip(np.dot(bvecs_non_b0.T, bvecs_non_b0), -1.0, 1.0))
        min_angle_seen = np.min(seen_angles[seen_angles > 0]) * 180 / np.pi
        max_angle_seen = np.max(seen_angles) * 180 / np.pi
        print(f'\nSeen directions angular coverage (excluding b0):')
        print(f'Minimum angle between directions: {min_angle_seen:.2f}°')
        print(f'Maximum angle between directions: {max_angle_seen:.2f}°')

    # Save files
    np.save(os.path.join(here, 'seen_directions.npy'), np.array(sub_idx))
    np.savetxt(sub_bvecs_txt, filtered_bvecs[:, sub_idx].T)
    np.savetxt(sub_bvals_txt, filtered_bvals[sub_idx])
    
    grad_mrtrix = np.column_stack((filtered_bvecs[:, sub_idx].T, filtered_bvals[sub_idx]))
    np.savetxt(os.path.join(here, 'grad.txt'), grad_mrtrix)

    dwi_sub = filtered_dwi[..., sub_idx]
    nib.save(nib.Nifti1Image(dwi_sub, nii_affine), sub_nii)

def convert_to_mif(here):
    """Convert DWI data from NIfTI to MRtrix (mif) format.

    This conversion is necessary for subsequent processing steps that use MRtrix tools.
    The function preserves the gradient information during conversion.

    Parameters
    ----------
    here : str
        Path to the directory containing input files and where output will be saved

    Notes
    -----
    Input files required:
    - dwi_sub.nii.gz
    - sub_bvecs.txt
    - sub_bvals.txt

    Output:
    - dwi.mif: DWI data in MRtrix format
    """
    if os.path.exists(os.path.join(here, 'dwi.mif')):
        print('dwi.mif already exists ✓')
        return
        
    mrconv = mrt.MRConvert()
    mrconv.inputs.in_file = os.path.join(here, 'dwi_sub.nii.gz')
    mrconv.inputs.grad_fsl = (os.path.join(here, 'sub_bvecs.txt'), os.path.join(here, 'sub_bvals.txt'))
    mrconv.inputs.out_file = os.path.join(here, 'dwi.mif')
    mrconv.run()

def calculate_wm_response(here, mask_nii):
    """Calculate the white matter response function for spherical deconvolution.

    This function estimates the signal response of a typical white matter fiber population,
    which is needed for the spherical deconvolution step. It uses the Tournier algorithm
    to estimate the response function.

    Parameters
    ----------
    here : str
        Path to the working directory
    mask_nii : str
        Path to the brain mask file

    Notes
    -----
    The function uses the 'tournier' algorithm, which is suitable for single-shell data.
    The response function is saved as 'response_wm.txt' and will be used in the
    subsequent CSD step.
    """
    if os.path.exists(os.path.join(here, 'response_wm.txt')):
        print('response_wm.txt already exists ✓')
        return
        
    resp = mrt.ResponseSD()
    resp.inputs.in_file = os.path.join(here, 'dwi.mif')
    resp.inputs.algorithm = 'tournier'
    resp.inputs.grad_fsl = (os.path.join(here, 'sub_bvecs.txt'), os.path.join(here, 'sub_bvals.txt'))
    resp.inputs.in_mask = mask_nii
    resp.inputs.wm_file = os.path.join(here, 'response_wm.txt')
    resp.run()

def perform_csd(here, mask_nii):
    """Perform Constrained Spherical Deconvolution (CSD) to estimate fiber orientations.

    CSD is used to estimate the fiber orientation distribution (FOD) from DWI data.
    This function deconvolves the measured DWI signal using the previously estimated
    white matter response function.

    Parameters
    ----------
    here : str
        Path to the working directory
    mask_nii : str
        Path to the brain mask file

    Notes
    -----
    The function requires:
    - The white matter response function (response_wm.txt)
    - The converted DWI data (dwi.mif)
    - Gradient information
    
    Output:
    - fod.mif: The estimated fiber orientation distribution
    """
    if os.path.exists(os.path.join(here, 'fod.mif')):
        print('fod.mif already exists ✓')
        return
        
    csd = mrt.ConstrainedSphericalDeconvolution()
    csd.inputs.algorithm = 'csd'
    csd.inputs.in_file = os.path.join(here, 'dwi.mif')
    csd.inputs.wm_txt = os.path.join(here, 'response_wm.txt')
    csd.inputs.wm_odf = os.path.join(here, 'fod.mif')
    csd.inputs.grad_fsl = (os.path.join(here, 'sub_bvecs.txt'), os.path.join(here, 'sub_bvals.txt'))
    csd.inputs.mask_file = mask_nii
    csd.run()

def synthesize_signal(here, bvals, bvecs, target_bval):
    """Re-synthesize the DWI signal from the estimated fiber orientation distribution.

    This function uses spherical harmonics to reconstruct the DWI signal for all
    original gradient directions, allowing comparison between the measured and
    synthesized signals.

    Parameters
    ----------
    here : str
        Path to the working directory
    bvals : ndarray
        Array of b-values
    bvecs : ndarray
        Array of gradient directions
    target_bval : int
        Target b-value used for filtering

    Notes
    -----
    The function generates:
    - orig_grad.txt: Original gradient directions in MRtrix format
    - dwi_synth.mif: Synthesized DWI signal in MRtrix format
    - dwi_synth.nii.gz: Synthesized DWI signal in NIfTI format
    """
    if os.path.exists(os.path.join(here, 'dwi_synth.nii.gz')):
        print('dwi_synth.nii.gz already exists ✓')
        return
        
    # Use the filtered bvals/bvecs that were used for CSD
    filtered_bvecs = np.loadtxt(os.path.join(here, 'sub_bvecs.txt')).T
    filtered_bvals = np.loadtxt(os.path.join(here, 'sub_bvals.txt'))
    
    # Save the gradient directions in MRtrix format
    grad_mrtrix = np.column_stack((filtered_bvecs.T, filtered_bvals))
    np.savetxt(os.path.join(here, 'orig_grad.txt'), grad_mrtrix)
    
    sh2amp = mrt.SH2Amp()
    sh2amp.inputs.in_file = os.path.join(here, 'fod.mif')
    sh2amp.inputs.directions = os.path.join(here, 'orig_grad.txt')
    sh2amp.inputs.out_file = os.path.join(here, 'dwi_synth.mif')
    sh2amp.run()

    mrc2 = mrt.MRConvert()
    mrc2.inputs.in_file = os.path.join(here, 'dwi_synth.mif')
    mrc2.inputs.out_file = os.path.join(here, 'dwi_synth.nii.gz')
    mrc2.run()

def evaluate_mse(here, dwi_nii, mask_nii, filtered_bvals, target_bval, bval_mask):
    """Evaluate the mean squared error between original and synthesized DWI data.

    This function compares the original and synthesized DWI signals, calculating
    separate MSE values for the seen (used in CSD) and unseen (held-out) directions.

    Parameters
    ----------
    here : str
        Path to the working directory
    dwi_nii : str
        Path to the original DWI data
    mask_nii : str
        Path to the brain mask
    filtered_bvals : ndarray
        Array of filtered b-values
    target_bval : int
        Target b-value used for filtering
    bval_mask : ndarray
        Boolean mask used for filtering b-values

    Returns
    -------
    tuple
        - orig_data (ndarray): Original DWI data
        - synth_data (ndarray): Synthesized DWI data
        - brain_mask (ndarray): Brain mask
        - seen_idx (ndarray): Indices of directions used in CSD
        - unseen_idx (ndarray): Indices of held-out directions

    Notes
    -----
    The MSE is calculated separately for seen and unseen directions to evaluate
    how well the FOD can predict the signal in new directions.
    """
    # Load original data (already filtered for target b-value)
    orig_data = nib.load(dwi_nii).get_fdata()[..., bval_mask]
    synth_data = nib.load(os.path.join(here, 'dwi_synth.nii.gz')).get_fdata()
    brain_mask = nib.load(mask_nii).get_fdata()

    # Load the indices of seen directions (these are relative to the filtered data)
    seen_idx = np.load(os.path.join(here, 'seen_directions.npy'))
    all_indices = np.arange(len(filtered_bvals))
    unseen_idx = np.array([i for i in all_indices if i not in seen_idx])

    print(f'\nEvaluating on {len(seen_idx)} seen directions and {len(unseen_idx)} unseen directions')

    brain_mask = brain_mask.astype(bool)
    min_signal = np.min(orig_data)
    max_signal = np.max(orig_data)
    orig_data_norm = (orig_data - min_signal) / (max_signal - min_signal)
    synth_data_norm = (synth_data - min_signal) / (max_signal - min_signal)

    brain_mask_4d = brain_mask[..., np.newaxis]
    
    # Ensure synth_data has the same number of volumes as orig_data
    if synth_data_norm.shape[-1] != orig_data_norm.shape[-1]:
        raise ValueError(f"Mismatch in number of volumes: original has {orig_data_norm.shape[-1]}, "
                       f"synthesized has {synth_data_norm.shape[-1]}")
    
    # Calculate MSE
    mse_seen = np.mean(((orig_data_norm[..., seen_idx] - synth_data_norm[..., seen_idx]) ** 2 * brain_mask_4d), axis=(0,1,2))
    mse_unseen = np.mean(((orig_data_norm[..., unseen_idx] - synth_data_norm[..., unseen_idx]) ** 2 * brain_mask_4d), axis=(0,1,2))
    
    print(f'MSE in brain mask (seen directions): {np.mean(mse_seen)}')
    print(f'MSE in brain mask (unseen directions): {np.mean(mse_unseen)}')
    
    return orig_data, synth_data, brain_mask, seen_idx, unseen_idx

def save_visualizations(here, orig_data, synth_data, brain_mask, seen_idx, unseen_idx):
    """Generate and save visualization plots comparing original and synthesized data.

    This function creates two types of visualizations:
    1. MSE maps showing the spatial distribution of errors
    2. Side-by-side comparisons of original and synthesized data for unseen directions

    Parameters
    ----------
    here : str
        Path to the working directory
    orig_data : ndarray
        Original DWI data
    synth_data : ndarray
        Synthesized DWI data
    brain_mask : ndarray
        Brain mask
    seen_idx : ndarray
        Indices of directions used in CSD
    unseen_idx : ndarray
        Indices of held-out directions

    Notes
    -----
    Outputs:
    - mse_comparison.png: Side-by-side MSE maps for seen and unseen directions
    - test/original_unseen_volX.png: Original data for each unseen direction
    - test/synthesized_unseen_volX.png: Synthesized data for each unseen direction
    """
    middle_slice = orig_data.shape[2] // 2
    
    # Save MSE maps
    voxel_mse_seen = np.mean(((orig_data[..., seen_idx] - synth_data[..., seen_idx]) ** 2), axis=-1)
    voxel_mse_unseen = np.mean(((orig_data[..., unseen_idx] - synth_data[..., unseen_idx]) ** 2), axis=-1)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(voxel_mse_seen[:, :, middle_slice] * brain_mask[:, :, middle_slice], cmap='hot')
    plt.colorbar()
    plt.title('MSE Map - Seen Directions')
    plt.subplot(122)
    plt.imshow(voxel_mse_unseen[:, :, middle_slice] * brain_mask[:, :, middle_slice], cmap='hot')
    plt.colorbar()
    plt.title('MSE Map - Unseen Directions')
    plt.savefig(os.path.join(here, 'mse_comparison.png'))
    plt.close()

    # Save middle slices
    test_dir = os.path.join(here, 'test')
    os.makedirs(test_dir, exist_ok=True)

    for idx, vol_idx in enumerate(unseen_idx):
        plt.figure(figsize=(10, 10))
        plt.imshow(orig_data[:, :, middle_slice, vol_idx], cmap='gray')
        plt.axis('off')
        plt.title(f'Original - Unseen Direction {idx + 1}')
        plt.savefig(os.path.join(test_dir, f'original_unseen_vol{idx + 1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.imshow(synth_data[:, :, middle_slice, vol_idx], cmap='gray')
        plt.axis('off')
        plt.title(f'Synthesized - Unseen Direction {idx + 1}')
        plt.savefig(os.path.join(test_dir, f'synthesized_unseen_vol{idx + 1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f'Middle slice images for unseen directions saved in {test_dir} ✓')

def main(args):
    """Main function to run the complete DWI analysis pipeline.

    This function orchestrates the entire process of:
    1. Loading and filtering DWI data
    2. Subsampling gradient directions
    3. Estimating fiber orientations using CSD
    4. Synthesizing DWI signal
    5. Evaluating and visualizing results

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments including:
        - root_dir: Path to input data directory
        - save_folder: Output directory name
        - target_bval: Target b-value for analysis

    Notes
    -----
    This is the main entry point for the analysis pipeline. It coordinates
    all the processing steps and ensures they are executed in the correct order.
    """
    # Setup
    here, dwi_nii, bvecs_txt, bvals_txt, mask_nii = setup_directories(args.save_folder, args.root_dir)
    
    # Filter b-value
    filtered_dwi, filtered_bvals, filtered_bvecs, nii_affine, bval_mask = filter_bvalue(
        dwi_nii, bvals_txt, bvecs_txt, args.target_bval
    )
    
    # Process data
    subsample_gradients(filtered_dwi, filtered_bvals, filtered_bvecs, here, nii_affine)
    convert_to_mif(here)
    calculate_wm_response(here, mask_nii)
    perform_csd(here, mask_nii)
    synthesize_signal(here, filtered_bvals, filtered_bvecs, args.target_bval)
    
    # Evaluate and visualize
    orig_data, synth_data, brain_mask, seen_idx, unseen_idx = evaluate_mse(
        here, dwi_nii, mask_nii, filtered_bvals, args.target_bval, bval_mask
    )
    save_visualizations(here, orig_data, synth_data, brain_mask, seen_idx, unseen_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Process DWI data and perform baseline directions analysis.
        
        This script performs the following steps:
        1. Filters DWI data for a specific b-value
        2. Subsamples gradient directions
        3. Performs CSD to estimate fiber orientations
        4. Synthesizes DWI signal
        5. Evaluates accuracy through MSE analysis
        
        Example usage:
        python baselines_directions.py --root_dir ../../HCP_example --save_folder results --target_bval 3000
        """
    )
    parser.add_argument('--root_dir', type=str, default='../../HCP_example',
                      help='Root directory containing input files (data.nii.gz, bvecs.txt, bvals.txt, brain_mask.nii.gz)')
    parser.add_argument('--save_folder', type=str, default='baselines_directions',
                      help='Folder name to save results')
    parser.add_argument('--target_bval', type=int, default=3000,
                      help='Target b-value for filtering (e.g., 1000, 2000, 3000)')
    
    args = parser.parse_args()
    main(args)

