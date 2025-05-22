import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import pandas as pd
from utils import load_data, split_datasets
from model import INRLightningModule
from dataset import DWIReprDataset
from val_dataset import ValidationCubeDataset
from feature_encoders import BvecAngleEncoder, SphericalPosEncoder
from logging_utils import compute_image_metrics, normalize_within_mask, init_lpips_model
from args_mappings import model_mapping, parse_args, parse_config


def load_model_from_checkpoint(config, train_data=None, val_data=None):
    """Load a trained model from checkpoint"""
    # Define network architecture based on config
    network_name = config.get('network_name', 'wire_mlp')  # Default to WIRE MLP for backward compatibility
    
    if network_name in model_mapping:
        inr = model_mapping[network_name](
            config['in_size'], 
            config['out_size'], 
            hidden_size=config['hidden_size'], 
            num_layers=config['num_layers']
        )
    else:
        raise ValueError(f"Unknown network name: {network_name}")
    
    # Set up position encoder based on network type
    use_pos_encoder = config.get('pos_encoder', True)  # Default to True for backward compatibility
    
    if use_pos_encoder:
        pos_encoder = [BvecAngleEncoder(7, (4, 5, 6)),
                   SphericalPosEncoder(l_max=7, 
                                       coord_size=4, 
                                       freq_num=64,
                                       freq_scale=1.
                                       )]
    else:
        pos_encoder = None

    evaluation_checkpoint = config['evaluation_checkpoint']
    # Get the maximum b-value from the dataset for proper unnormalization
    data_max_bval = train_data.bvals.max()
    max_bval = config.get('max_bval', data_max_bval) if config.get('max_bval') else data_max_bval
    val_dataloaders = {'slice': val_data} 
    # Load model from fixed checkpoint
    pl_module = INRLightningModule.load_from_checkpoint(
        evaluation_checkpoint,
        network=inr,
        lr=config['lr'],
        val_slice_func=None,  # Will be set later
        log_coords_raw=config.get('log_coords_raw', False),
        log_coords_encoded=config.get('log_coords_encoded', False),
        eval_gradients=config.get('eval_gradients', [0]),
        name=config.get('model_name', ''),
        max_bval=max_bval,
        steps_per_viz=config.get('steps_per_viz', 2000),
        train_loader=train_data,
        val_loaders_dict=val_dataloaders
    )
    
    pl_module.eval()
    return pl_module


def create_datasets(config, dwi, mask, bvec, bval, header=None):
    """Create datasets based on config"""
    # Set default values for normalization if not specified in config
    normalize_0_1 = config.get('normalize_0_1', False)
    normalize_in_out = config.get('normalize_in_out', True)  # Default for backward compatibility
    
    print(f"normalize_0_1: {normalize_0_1}, normalize_in_out: {normalize_in_out}")
    
    # Get max_bval from config for proper normalization and unnormalization
    max_bval = config.get('max_bval', 3000)
    print(f"Using max_bval: {max_bval} for normalization")
    
    # Split datasets if train_split and val_split are provided
    train_split = config.get('train_split', None)
    val_split = config.get('val_split', None)
    if train_split is not None and val_split is not None:
        print(f"Splitting dataset: train={train_split}, val={val_split} stratified by b-values")
        (train_dwi, train_bvec, train_bval), \
        (val_dwi, val_bvec, val_bval) = split_datasets(dwi, bvec, bval, 
                                                      train_split=train_split,
                                                      val_split=val_split)
    else:
        print("Using full dataset for both training and validation")
        train_dwi, train_bvec, train_bval = dwi, bvec, bval
        val_dwi, val_bvec, val_bval = dwi, bvec, bval
    
    train_dataset = DWIReprDataset(
        train_dwi, mask, train_bvec, train_bval, 
        normalize_0_1=normalize_0_1, 
        normalize_in_out=normalize_in_out, 
        bval_selection=config.get('bval_train'),
        max_bval=max_bval,  # Explicitly pass max_bval
        is_train=False,
        dwi_headers=header
    )
    
    val_dataset = DWIReprDataset(
        val_dwi, mask, val_bvec, val_bval, 
        normalize_0_1=normalize_0_1, 
        normalize_in_out=normalize_in_out, 
        bval_selection=config.get('bval_val'),
        max_bval=max_bval,  # Explicitly pass max_bval 
        is_train=False
    )
    
    return train_dataset, val_dataset


def evaluate_complete_volume(model, datasets, config, device='cuda'):
    """Evaluate model on the entire volume across all slices and b-values"""
    model.to(device)
    model.eval()
    train_dataset, val_dataset = datasets
    
    # Initialize LPIPS model once here
    lpips_model = init_lpips_model(device)
    print("LPIPS model initialized once for all evaluations")
    
    # Get volume dimensions from the validation dataset
    volume_shape = val_dataset.image.shape[:3]  # (x, y, z)
    
    print(f"\nEvaluating on validation split:")
    print(f"Total b-values in validation: {len(val_dataset.bvals_val)}")
    print(f"B-values: {val_dataset.bvals_val.tolist()}")
    
    # Get max_bval for proper unnormalization
    max_bval = config.get('max_bval', 3000)
    print(f"Using max_bval: {max_bval} for unnormalization")
    
    # Create a validation dataset with return_volumes=True
    validation_dataset = ValidationCubeDataset(
        val_dataset,
        normalize_in_out=config.get('normalize_in_out', True),
        normalize_0_1=config.get('normalize_0_1', False),
        return_volumes=True
    )
    
    # Print the actual b-values from the validation dataset for debugging
    print("\nAvailable b-values in validation dataset:")
    for i, bval in enumerate(validation_dataset.bvals_val):
        rounded_bval = round(bval.item() / 100) * 100
        print(f"  Index {i}: {bval.item()} (rounded to {rounded_bval})")
    
    # Create dataloader for the validation dataset
    dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # Set the val_slice_func for visualization
    model.val_slice_func = validation_dataset._fold_slice
 
    # Initialize metrics storage per b-value
    bvalue_metrics = {}
    dwi_pred = torch.tensor([])
    dwi_gt = torch.tensor([])
    # Run evaluation on all slices
    with torch.no_grad():
        # Initialize storage for all slices per b-value
        volume_predictions = {}
        volume_targets = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating volumes")):
            x, y, idx = batch
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Calculate b-value index
            idx_val = idx.item()
            b_idx = idx_val 
            
            # Get actual b-value
            actual_bval = validation_dataset.bvals_val[b_idx].item()
            
            # Round b-value to nearest 100
            rounded_bval = round(actual_bval / 100) * 100
            
            # Initialize storage for this b-value if not already present
            if rounded_bval not in bvalue_metrics:
                bvalue_metrics[rounded_bval] = {
                    'mse': [], 'mae': [], 'relative_error': [], 
                    'psnr': [], 'ssim': [], 'lpips': []
                }
            
            # Forward pass
            y_hat = model(x)
            brain_mask = validation_dataset.reference_dataset.mask.bool().to(y_hat.device)
            # Get the mask for this 
            mask = validation_dataset.reference_dataset.mask.bool()
         
            if mask.device != y_hat.device:
                mask = mask.to(y_hat.device)
            
            # Reshape predictions and targets to match the slice dimensions
            y_hat_volume = y_hat.reshape(validation_dataset.reference_dataset.dim[:3])
            y_volume = y.reshape(validation_dataset.reference_dataset.dim[:3])
            
            # Denormalize the data if normalization was used
            if (validation_dataset.normalize_in_out or validation_dataset.normalize_0_1) and not config['no_denorm']:
                print("Denormalizing the data")
                # Create tensors with the right shape for denormalization
                x_dummy = x[:1]  # Just need one sample for denormalization
                # Reshape to match the expected input format for denormalization
                y_hat_to_denorm = y_hat.reshape(-1, 1)
                y_to_denorm = y.reshape(-1, 1)
                # Apply appropriate denormalization based on the normalization method
                if validation_dataset.normalize_0_1:
                    # Make sure we're using the same reference dataset with correct max_bval
                    _, y_hat_denorm = validation_dataset.reference_dataset.normalize_xy_01(
                        x_dummy, y_hat_to_denorm, inverse=True)
                    _, y_denorm = validation_dataset.reference_dataset.normalize_xy_01(
                        x_dummy, y_to_denorm, inverse=True)
                else:
                    # Make sure we're using the same reference dataset with correct max_bval
                    _, y_hat_denorm = validation_dataset.reference_dataset.normalize_xy_meanstd(
                        x_dummy, y_hat_to_denorm, inverse=True)
                    _, y_denorm = validation_dataset.reference_dataset.normalize_xy_meanstd(
                        x_dummy, y_to_denorm, inverse=True)
                
                # Print the max_bval used for denormalization for debugging
                print(f"Using max_bval: {validation_dataset.reference_dataset.max_bval} for denormalization")
                
                # Reshape back to volume shape
                y_hat_volume = y_hat_denorm.reshape(validation_dataset.reference_dataset.dim[:3])
                y_volume = y_denorm.reshape(validation_dataset.reference_dataset.dim[:3])
            
            # Apply mask to get only brain voxels
            y_hat_masked = y_hat_volume[brain_mask]
            y_masked = y_volume[brain_mask]
                
            # Calculate metrics on masked volume
            mse = torch.nn.functional.mse_loss(y_hat_masked, y_masked).item()
            mae = torch.nn.functional.l1_loss(y_hat_masked, y_masked).item()

          
            # Calculate relative error with increased epsilon and clamping
            epsilon = 1e-4
            y_safe = torch.clamp(torch.abs(y_masked), min=epsilon)
            rel_error = torch.mean(torch.abs(y_hat_masked - y_masked) / y_safe).item() * 100
            rel_error = min(rel_error, 1000.0)  # Clamp to prevent inf

            # apply mask to the volume
            y_volume_masked = y_volume*brain_mask
            y_hat_volume_masked = y_hat_volume*brain_mask
            dwi_pred = torch.cat((dwi_pred, y_hat_volume_masked.unsqueeze(-1).detach().cpu()), dim=-1)
            dwi_gt = torch.cat((dwi_gt, y_volume_masked.unsqueeze(-1).detach().cpu()), dim=-1)

                
            psnr_val, ssim_val, lpips_val = compute_image_metrics(
                y_hat_volume, y_volume, device=device, lpips_model=lpips_model, mask=brain_mask
            )
           
            # Store metrics for this b-value
            bvalue_metrics[rounded_bval]['mse'].append(mse)
            bvalue_metrics[rounded_bval]['mae'].append(mae)
            bvalue_metrics[rounded_bval]['relative_error'].append(rel_error)
            bvalue_metrics[rounded_bval]['psnr'].append(psnr_val)
            bvalue_metrics[rounded_bval]['ssim'].append(ssim_val)
            bvalue_metrics[rounded_bval]['lpips'].append(lpips_val)
    

    # Convert metrics to DataFrame format
    all_metrics = {
        'mse': [], 'mae': [], 'relative_error': [], 
        'psnr': [], 'ssim': [], 'lpips': [], 'b_value': []
    }
    
    # Calculate average metrics for each b-value
    for b_val, metrics in bvalue_metrics.items():
        for metric_key in metrics:
            print(f"Number of values for {metric_key} and b-value {b_val}: {len(metrics[metric_key])}")
            avg_value = np.mean(metrics[metric_key])
            all_metrics[metric_key].append(avg_value)
        all_metrics['b_value'].append(b_val)
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    return metrics_df, dwi_pred, dwi_gt


def save_evaluation_results(metrics_df, output_dir, config):
    """Save evaluation results to CSV and generate summary plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw metrics
    metrics_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Generate summary by b-value
    bvalue_summary = metrics_df.groupby('b_value').mean().reset_index()
    bvalue_summary.to_csv(os.path.join(output_dir, 'metrics_summary_by_bvalue.csv'), index=False)
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    
    # Plot MSE by b-value
    plt.subplot(2, 2, 1)
    plt.bar(bvalue_summary['b_value'].astype(str), bvalue_summary['mse'])
    plt.title('MSE by b-value')
    plt.xlabel('b-value')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    
    # Plot Relative Error by b-value
    plt.subplot(2, 2, 2)
    plt.bar(bvalue_summary['b_value'].astype(str), bvalue_summary['relative_error'])
    plt.title('Relative Error (%) by b-value')
    plt.xlabel('b-value')
    plt.ylabel('Relative Error (%)')
    plt.xticks(rotation=45)
    
    # Plot PSNR by b-value
    plt.subplot(2, 2, 3)
    plt.bar(bvalue_summary['b_value'].astype(str), bvalue_summary['psnr'])
    plt.title('PSNR by b-value')
    plt.xlabel('b-value')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45)
    
    # Plot SSIM by b-value
    plt.subplot(2, 2, 4)
    plt.bar(bvalue_summary['b_value'].astype(str), bvalue_summary['ssim'])
    plt.title('SSIM by b-value')
    plt.xlabel('b-value')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_bvalue.png'))
    
    # Calculate overall metrics
    overall = metrics_df.mean()
    with open(os.path.join(output_dir, 'overall_metrics.txt'), 'w') as f:
        f.write(f"Overall MSE: {overall['mse']:.6f}\n")
        f.write(f"Overall MAE: {overall['mae']:.6f}\n")
        f.write(f"Overall Relative Error (%): {overall['relative_error']:.2f}\n")
        f.write(f"Overall PSNR (dB): {overall['psnr']:.2f}\n")
        f.write(f"Overall SSIM: {overall['ssim']:.4f}\n")
        f.write(f"Overall LPIPS: {overall['lpips']:.4f}\n")
    
    return bvalue_summary, None  # Return None for volume_summary since we no longer track per-volume metrics


def main():
    args = parse_args(evaluation_mode=True)
    config = parse_config(args.config, subject=args.subject, rootdir=args.rootdir)
    output_dir = args.output_dir
    config['no_denorm'] = args.no_denorm

    # Check if evaluation checkpoint path is provided
    if 'evaluation_checkpoint' not in config or not config['evaluation_checkpoint']:
        print("Error: No evaluation checkpoint path provided in config file")
        sys.exit(1)
    print("Loading model from checkpoint: ", config["evaluation_checkpoint"]) 
    # Load data
    print("Loading data...")
    dwi, mask, bvec, bval, header = load_data(config, return_header=True)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset, val_dataset = create_datasets(config, dwi, mask, bvec, bval, header)
    
    # Load model
    print(f"Loading model from {config['evaluation_checkpoint']}...")
    model = load_model_from_checkpoint(config, train_data=train_dataset, val_data=val_dataset)
    
    # Run volume evaluation
    print("Running volume-based evaluation...")
    # metrics_df = evaluate_volume(model, (train_dataset, val_dataset), config)
    metrics_df, dwi_pred, dwi_gt = evaluate_complete_volume(model, (train_dataset, val_dataset), config)
    # save the predicted and ground truth volumes as nifti files
    if not args.no_save_volumes:
        dwi_pred_nii = nib.Nifti1Image(dwi_pred.numpy(), None, header=header)
        dwi_gt_nii = nib.Nifti1Image(dwi_gt.numpy(), None, header=header)
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
