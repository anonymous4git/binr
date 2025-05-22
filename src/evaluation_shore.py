import torch
import numpy as np
import nibabel as nib
import os
import pandas as pd
from utils import load_data, split_datasets
from evaluation import create_datasets, evaluate_complete_volume, save_evaluation_results
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import ShoreModel, shore_matrix
from logging_utils import compute_image_metrics, normalize_within_mask, init_lpips_model
from agrs_mappings import parse_args, parse_config


class ShoreModelEvaluator:
    def __init__(self, radial_order=6, zeta=700, lambdaN=1e-8, lambdaL=1e-8, b0_threshold=10):
        self.radial_order = radial_order
        self.zeta = zeta
        self.lambdaN = lambdaN
        self.lambdaL = lambdaL
        self.b0_threshold = b0_threshold
        self.model = None
        self.S0 = None
        self.fit_result = None
        self.mask = None
        self.masked_shape = None

    def fit(self, dwi_data, bvals, bvecs, mask=None):
        # Set the mask if provided
        if mask is not None:
            self.mask = mask
            dwi_data = dwi_data[mask]
            self.masked_shape = dwi_data.shape[:-1]
        
        # Ensure gradient table uses keyword arguments
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        b0_mask = np.abs(gtab.bvals) <= self.b0_threshold
        self.S0 = np.mean(dwi_data[..., b0_mask], axis=-1)
        self.model = ShoreModel(gtab, radial_order=self.radial_order, zeta=self.zeta, lambdaN=self.lambdaN, lambdaL=self.lambdaL)
        self.fit_result = self.model.fit(dwi_data)

    def predict(self, bvals, bvecs):
        if self.model is None or self.S0 is None:
            raise ValueError("Model must be fitted before predicting.")
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        M = shore_matrix(
            radial_order=self.model.radial_order,
            zeta=self.model.zeta,
            gtab=gtab,
            tau=self.model.tau
        )
        # Predict and reshape to original volume if mask is present
        if self.mask is not None and self.masked_shape is not None:
            # Use vectorized multiplication for masked data
            pred_signal = np.zeros(self.mask.shape + (M.shape[0],))
            masked_signal = np.einsum("df,nf->nd", M, self.fit_result.shore_coeff) * self.S0[..., None].reshape(-1, 1)
            pred_signal[self.mask] = masked_signal

            # replace NAN with 0
            pred_signal[np.isnan(pred_signal)] = 0.

            return pred_signal
        else:
            return np.einsum("df,ijkf->ijkd", M, self.fit_result.shore_coeff) * self.S0[..., None]


def compute_metrics(y_volume, y_hat_volume, brain_mask, bvals, device):
    def bvals_to_list(bv):
        if isinstance(bv, np.ndarray) or isinstance(bv, list):
            bv = [int(round(b)) for b in bv]
        elif isinstance(bv, torch.Tensor):
            bv = [int(round(b.item())) for b in bv]
        else:
            raise TypeError(f"Unsupported type for bvals: {type(bv)}")

        return bv

    bvals = bvals_to_list(bvals)

    # Initialize LPIPS model once
    lpips_model = init_lpips_model(device)
    print("LPIPS model initialized once for all evaluations")
    
    bvalue_metrics = {}
    
    # Extract and compute metrics per b-value
    for idx, actual_bval in enumerate(bvals):
        # Round b-value to nearest 100
        rounded_bval = round(actual_bval / 100) * 100
        
        # Apply mask to get only brain voxels
        y_hat_masked = y_hat_volume[brain_mask][..., idx]
        y_masked = y_volume[brain_mask][..., idx]
        
        # Calculate basic metrics
        mse = torch.nn.functional.mse_loss(y_hat_masked, y_masked).item()
        mae = torch.nn.functional.l1_loss(y_hat_masked, y_masked).item()
        
        # Calculate relative error with clamping
        epsilon = 1e-4
        y_safe = torch.clamp(torch.abs(y_masked), min=epsilon)
        rel_error = torch.mean(torch.abs(y_hat_masked - y_masked) / y_safe).item() * 100
        rel_error = min(rel_error, 1000.0)
        
        # Compute additional metrics
        psnr_val, ssim_val, lpips_val = compute_image_metrics(
            y_hat_volume[..., idx], y_volume[..., idx], device=device, lpips_model=lpips_model, mask=brain_mask
        )
        
        # Store metrics for this b-value
        if rounded_bval not in bvalue_metrics:
            bvalue_metrics[rounded_bval] = {'mse': [], 'mae': [], 'relative_error': [], 'psnr': [], 'ssim': [], 'lpips': []}
        bvalue_metrics[rounded_bval]['mse'].append(mse)
        bvalue_metrics[rounded_bval]['mae'].append(mae)
        bvalue_metrics[rounded_bval]['relative_error'].append(rel_error)
        bvalue_metrics[rounded_bval]['psnr'].append(psnr_val)
        bvalue_metrics[rounded_bval]['ssim'].append(ssim_val)
        bvalue_metrics[rounded_bval]['lpips'].append(lpips_val)
    
    # Convert metrics to DataFrame
    all_metrics = {'mse': [], 'mae': [], 'relative_error': [], 'psnr': [], 'ssim': [], 'lpips': [], 'b_value': []}
    for b_val, metrics in bvalue_metrics.items():
        for metric_key in metrics:
            avg_value = np.mean(metrics[metric_key])
            all_metrics[metric_key].append(avg_value)
        all_metrics['b_value'].append(b_val)
    
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


def normalize_y_01(y, inverse=False, y_min=None, y_max=None):
    # Compute min and max for normalization
    y_min = y.min() if y_min == None else y_min
    y_max = y.max() if y_max == None else y_max
    
    # Normalize y
    if not inverse:
        y_n = (y - y_min) / (y_max - y_min)
    else:
        y_n = (y * (y_max - y_min)) + y_min
    return y_n


def main():
    args = parse_args(evaluation_mode=True)
    config = parse_config(args.config, subject=args.subject, rootdir=args.rootdir)
    output_dir = args.output_dir
    config['no_denorm'] = args.no_denorm
    
    # Load data
    print("Loading data...")
    dwi, mask, bvec, bval = load_data(config)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset, val_dataset = create_datasets(config, dwi, mask, bvec, bval)
    
    # TODO REMOVE!! (args for faster testing)
    # print('before debug fitting')
    # sme = ShoreModelEvaluator(radial_order=2, zeta=700, lambdaN=1e-6, lambdaL=1e-6)
    # print('after debug fitting')
    # TODO enradial_order=6, zeta=700, lambdaN=1e-6, lambdaL=1e-6d

    sme = ShoreModelEvaluator()
    im = train_dataset.image.numpy()
    bve = train_dataset.bvecs.numpy()
    bva = train_dataset.bvals.numpy()
    mask = train_dataset.mask.numpy()

    # FIT SHORE    
    sme.fit(im, bva, bve, mask)

    # PREDICT SHORE
    device='cuda'

    dwi_pred = torch.tensor(sme.predict(val_dataset.bvals.numpy(), val_dataset.bvecs.numpy()), device=device)
    dwi_gt = val_dataset.image.to(device=device)
    mask_val = val_dataset.mask.to(device=device)

    # import ipdb; ipdb.set_trace()

    # NORMALIZE (to be consistent)
    if val_dataset.normalize_0_1 and config['no_denorm']:
        print("Normalizing the data")
        dwi_pred = normalize_y_01(dwi_pred, 
                                  y_min=val_dataset.y_min,
                                  y_max=val_dataset.y_max)
        dwi_gt = normalize_y_01(dwi_gt,
                                y_min=val_dataset.y_min,
                                y_max=val_dataset.y_max)
        
    elif not config['no_denorm']:
        print("Dwi already de_normed, no need to do anything")
    else:
        raise NotImplementedError

    print("Running volume-based evaluation...")
    metrics_df = compute_metrics(dwi_gt, dwi_pred, mask_val, val_dataset.bvals.to(device=device), device)

    # save the predicted and groprior_dw_schemeund truth volumes as nifti files
    if not args.no_save_volumes:
        dwi_pred_nii = nib.Nifti1Image(dwi_pred.detach().to(float).detach().cpu().numpy(), np.eye(4))
        dwi_gt_nii = nib.Nifti1Image(dwi_gt.detach().to(float).cpu().numpy(), np.eye(4))
        dwi_path = os.path.join(output_dir, 'dwi')
        os.makedirs(dwi_path, exist_ok=True)
        nib.save(dwi_pred_nii, os.path.join(dwi_path, 'dwi_pred.nii.gz'))
        nib.save(dwi_gt_nii, os.path.join(dwi_path, 'dwi_gt.nii.gz'))
        # save bvec and bval files
        np.savetxt(os.path.join(dwi_path, 'bvecs.txt'), bvec)
        np.savetxt(os.path.join(dwi_path, 'bvals.txt'), bval)

    # Save results
    print(f"Saving results to {output_dir}...")
    bvalue_summary, _ = save_evaluation_results(metrics_df, output_dir, config)
    
    print("Evaluation complete!")
    print("\nSummary of metrics by b-value:")
    print(bvalue_summary)
    
    # Print overall metrics
    overall = metrics_df.mean()
    print("\nOverall metrics:")
    print(f"MSE: {overall['mse']:.6f}")
    print(f"MAE: {overall['mae']:.6f}")
    print(f"Relative Error (%): {overall['relative_error']:.2f}")
    print(f"PSNR (dB): {overall['psnr']:.2f}")
    print(f"SSIM: {overall['ssim']:.4f}")
    print(f"LPIPS: {overall['lpips']:.4f}")


if __name__ == "__main__":
    main()
