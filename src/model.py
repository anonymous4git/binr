import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from logging_utils import compute_image_metrics, normalize_within_mask, log_metrics_and_images, log_multirow_bvalue_visualization
import nibabel as nib
import os


class INRLightningModule(pl.LightningModule):
    def __init__(
        self,
        network,
        train_loader, 
        val_loaders_dict,
        pos_encoder=None,
        lr=0.001,
        weight_decay=0.0,
        name="",
        val_slice_func=None,
        log_coords_raw=False,
        log_coords_encoded=False,
        eval_gradients=None,
        max_bval=None,
        steps_per_viz=2000,
        export_dwi=False,
        epochs=1000,
        use_cos_lrsheduler=False,
        train_log_interval=1,
        use_dropout=False,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cos_lrsheduler=use_cos_lrsheduler
        self.epochs=epochs
        self.network = network
        self.pos_encoder = self.setup_pos_encoder(pos_encoder)
        self.name = name
        self.val_slice_func = val_slice_func
        self.log_coords_raw = log_coords_raw
        self.log_coords_encoded = log_coords_encoded
        self.export_dwi=export_dwi
        self.use_dropout = use_dropout
        # attach dataloaders here - required to run multible
        self._train_loader = train_loader
        if val_loaders_dict is not None:
            self._val_loader_dict = val_loaders_dict
            self._val_loader_names = list(val_loaders_dict.keys())

        # Add gradient indices to evaluate - default to [0] if none provided
        self.eval_gradients = eval_gradients if eval_gradients is not None else [0]
        self.max_bval = max_bval
        self.steps_per_viz = steps_per_viz
        self.epsilon = 1e-8
        
        # Save hyperparameters for checkpoint loading, but exclude functions
        self.save_hyperparameters(ignore=['val_slice_func', 'train_slice_func', 'val_loaders_dict', 'train_loader'])

    def log(self, name, value, *args, **kwargs):
        """
        wrapper to log only in an interval during training
        """        
        kwargs.setdefault("on_step", False)
        kwargs.setdefault("on_epoch", True)
        # if self.training:
        #     if self.global_step % self.train_log_interval != 0:
        #         return  # skip logging this step
        super().log(name, value, *args, **kwargs)

    @staticmethod
    def setup_pos_encoder(pos_encoder):
        if pos_encoder is not None:
            if isinstance(pos_encoder, torch.nn.Module):
                pe = torch.nn.ModuleList([pos_encoder])
            else:
                pe = torch.nn.ModuleList(pos_encoder)
        else:
            pe = None

        return pe

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        if self.use_cos_lrsheduler:
            # Set min LR to 1% of initial
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.epochs, 
                eta_min=self.lr * 0.01
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or 'step' if you want per-step updates
                    "frequency": 1,
                    "monitor": None
                }
            }

        else:
            return optimizer

    
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        # return list here such that we can acess different dataloaders
        return list(self._val_loader_dict.values())

    def forward(self, coords):
        # Apply dropout to bvalues during training
        if self.training and self.use_dropout:
            # Get the original shape to restore later
            orig_shape = coords.shape
            
            # Extract bvalues which are at index 3
            bvals = coords[:, 3:4]

            # add a 20 % perturbation to the bvalues
            #bvals = bvals + (torch.rand_like(bvals)-0.5) * 0.2 * bvals
            
            # Apply dropout to bvalues with probability 0.2
            dropout_mask = torch.rand_like(bvals) > 0.5
            coords = coords.clone()  # Create a copy to avoid modifying the original
            coords[:, 3:4] = bvals * dropout_mask  # Zero out some bvalues

        if self.pos_encoder is not None:
            for encoder in self.pos_encoder:
                coords = encoder(coords)

        if self.log_coords_encoded:
            for i in range(coords.shape[-1]):
                self.logger.experiment.add_histogram(
                    f"encoded_coords_{i}", coords[:, i], self.global_step)

        return self.network(coords)

    def training_step(self, batch, batch_idx):

        coords, values, _ = batch
        coords = coords.view(-1, coords.shape[-1])
        values = values.view(-1, values.shape[-1])
        outputs = self.forward(coords)
        # Ensure the shapes are consistent
        # outputs = outputs.view_as(values)
        
        loss = F.mse_loss(outputs, values)

        # logging
        self.log(f"{self.name}/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr)

        # Calculate and log relative error (MAPE) with increased epsilon
        with torch.no_grad():
            # Also clip very small values to reduce extreme relative errors
            relative_error = torch.mean(
                torch.abs(outputs - values) / (torch.abs(values) + 1e-3)
            ) * 100            
            
            self.log(f"{self.name}/train_relative_error_%", relative_error, on_step=False, prog_bar=True)

            # If the data has b-values (for diffusion MRI), compute per-bvalue errors
            if coords.shape[-1] > 4:  # Assuming the last column might contain b-values
                bvals = coords[:, 3]  # Last column contains normalized b-values
                unique_bvals = torch.unique(bvals) 
                
                # Show both normalized and unnormalized b-values for clarity
                # if self.max_bval is not None:
                #     unnormalized_bvals = [b.item() * self.max_bval for b in unique_bvals]
                #     rounded_unnorm = [round(b.item() / 100) * 100 for b in unnormalized_bvals]
                # else:
                #     rounded_unique = [round(b.item() / 100) * 100 for b in unique_bvals]
                
                for bval in unique_bvals:
                    # Use a small epsilon for floating point comparison
                    mask = torch.isclose(bvals, bval, rtol=1e-5)
                    if mask.any():
                        values_masked_safe = torch.clamp(torch.abs(values[mask]), min=self.epsilon)
                        bval_error = torch.mean(
                            torch.abs(outputs[mask] - values[mask]) / values_masked_safe
                        ) * 100
                        
                        # Unnormalize and round to closest 100 for logging
                        if self.max_bval is not None:
                            actual_bval = bval.item() * self.max_bval # .item()
                            rounded_bval = round(actual_bval / 100) * 100
                        else:
                            # Fallback if max_bval not provided
                            rounded_bval = round(bval.item() / 100) * 100
                            
                        self.log(f"{self.name}/relative_error_b{rounded_bval:.0f}", bval_error)

        # logging - histogram - input
        # if self.log_coords_raw:
        #     self.logger.experiment.add_histogram("in[:3]", coords[:, :3], self.global_step)
        #     self.logger.experiment.add_histogram("in[3]", coords[:, 3], self.global_step)
        #     self.logger.experiment.add_histogram("in[4:]", coords[:, 4:], self.global_step)

        # logging - histogram - ouput
        # self.logger.experiment.add_histogram("out_true", values, self.global_step)
        # self.logger.experiment.add_histogram("out_pred", outputs, self.global_step)

        return loss
    
    @staticmethod
    def normalize_for_tensorboard(img):
        """Normalize tensor to [0,1] for TensorBoard logging"""
        img_min = img.min()
        img_max = img.max()

        return (img - img_min) / (img_max - img_min + 1e-8)  # Avoid division by zero

    def on_after_backward(self):
        grads = [p.grad.norm().item() for p in self.network.parameters() if p.grad is not None]
        self.log(f"{self.name}/grad_norm", sum(grads) / len(grads))

    def get_bval_from_idx(self, b_idx, val_type):

        loader = self._val_loader_dict[val_type]
    
        # Get the b-value identifier for logging
        b_value = loader.dataset.b_indices[b_idx]
        if hasattr(loader.dataset.reference_dataset, 'bvals'):
            actual_bval = loader.dataset.reference_dataset.bvals[b_value].item()
            rounded_bval = round(actual_bval / 100) * 100
        else:
            rounded_bval = b_value

        return rounded_bval

    def validate_slice(self, x_list, y_list):
        # Initialize metrics for averaging across b-values (better: do this for volume)
        total_loss = 0
        total_rel_error = 0
         
         # Process each b-value separately
        for b_idx, (x, y) in enumerate(zip(x_list, y_list)):

            # SLICE METRICS

            # Forward pass for this b-value
            
            y_hat = self.forward(x)

            y_slice = self.normalize_for_tensorboard(self.val_slice_func(x, y, denormalize=True, b_idx_pos=b_idx))
            y_hat_slice = self.normalize_for_tensorboard(self.val_slice_func(x, y_hat.squeeze(), denormalize=True, b_idx_pos=b_idx))
            se_slice = (y_slice - y_hat_slice) ** 2
                    
            # Compute image quality metrics
            psnr_val, ssim_val, lpips_val = compute_image_metrics(
                y_hat_slice, y_slice, device=self.device
            )
                    
            rounded_bval = self.get_bval_from_idx(b_idx=b_idx, val_type='slice')

            # Log metrics for this b-value
            self.log(f"{self.name}/val_psnr_b{rounded_bval}", psnr_val)
            self.log(f"{self.name}/val_ssim_b{rounded_bval}", ssim_val)
            self.log(f"{self.name}/val_lpips_b{rounded_bval}", lpips_val)
                    
            # Log images for this b-value
            self.logger.experiment.add_image(
                f"{self.name}/val_target_b{rounded_bval}",
                y_slice.unsqueeze(0),
                self.global_step
            )
            self.logger.experiment.add_image(
                f"{self.name}/val_prediction_b{rounded_bval}",
                y_hat_slice.unsqueeze(0),
                self.global_step
            )
            self.logger.experiment.add_image(
                f"{self.name}/val_squared_error_b{rounded_bval}",
                se_slice.unsqueeze(0),
                self.global_step
            )

            # GENERAL METRICS, evaluated on slice (for now)
            # Calculate loss
            loss = F.mse_loss(y_hat, y)
            total_loss += loss
                
            # Calculate relative error with increased epsilon and clamping
            epsilon = 1e-8
            y_safe = torch.clamp(torch.abs(y), min=epsilon)
            rel_error = torch.mean(torch.abs(y_hat - y) / y_safe) * 100
            rel_error = torch.clamp(rel_error, max=1000.0)
            total_rel_error += rel_error
                
            # Log individual b-value metrics
            self.log(f"{self.name}/val_loss_b{rounded_bval}", loss)
            self.log(f"{self.name}/val_relative_error_b{rounded_bval}_%", rel_error)
                
           
        # Calculate and log average metrics across b-values
        num_b_values = len(x_list)
        avg_loss = total_loss / num_b_values
        avg_rel_error = total_rel_error / num_b_values
            
        self.log(f"{self.name}/val_loss", avg_loss)
        self.log(f"{self.name}/val_relative_error_%", avg_rel_error)

        return avg_loss
 


    def validate_volume(self, x_, y_):

        if self.export_dwi:
            # Process all b-values together (that are included in this volume)
            y_hat = self.forward(x_)

            dset = self._val_loader_dict['volume'].dataset             

            # fold ...
            dwi = dset._fold_xyz(x_, y_, skip_checks=True, denormalize=True, force_int_coords=True)
            dwi_hat = dset._fold_xyz(x_, y_hat, skip_checks=True, denormalize=True, force_int_coords=True) # bit hacky with the int coords

            # dump to disk
            nib.save(nib.Nifti1Image(dwi.cpu().numpy().astype(float), affine=np.eye(4)), 
                     os.path.join(self.logger.log_dir, 'dwi.nii.gz'))
            nib.save(nib.Nifti1Image(dwi_hat.cpu().numpy().astype(float), affine=np.eye(4)), 
                     os.path.join(self.logger.log_dir, 'dwi_hat.nii.gz'))

            # also save bvecs and bvals here bcause why not
            bvecs = dset.bvecs.T 
            np.savetxt(os.path.join(self.logger.log_dir, "bvecs"), 
                       bvecs, fmt="%.5f", delimiter=" ")
            bvals = dset.bvals
            np.savetxt(os.path.join(self.logger.log_dir, "bvals"), 
                       bvals[np.newaxis, :], fmt="%d", delimiter=" ")

        else:
            return # Do nothing for now


    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # fetch type of dataloader 
        val_type = self._val_loader_names[dataloader_idx]
        
        # Unpack the batch - now x and y are lists of tensors, one per b-value
        x_list, y_list, idx = batch
        
        # case 1: just logging via slice dataloader
        # Generate and log slice visualizations for this b-value
        if self.val_slice_func is not None and val_type=='slice':
            self.validate_slice(x_list, y_list)

            return

        # case 2: evaluate on a volume - no logging of img to tb!
        elif val_type=='volume':
            avg_loss = self.validate_volume(x_list, y_list)

            return avg_loss

        else:
            raise ValueError('unrecognized validation dataloader type, pick <slice, volume>')
