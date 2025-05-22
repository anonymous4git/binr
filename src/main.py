import torch
from torch.utils.data import DataLoader
import sys
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from agrs_mappings import model_mapping, parse_args, parse_config
from utils import load_data, split_datasets
from model import INRLightningModule
from dataset import DWIReprDataset, PreBatchedDWIReprDataset
from val_dataset import ValidationSliceDataset
from networks.pos_encoders import BvecAngleEncoder, SphericalPosEncoder
# from pytorch_lightning.profilers import PyTorchProfiler
import os
torch.set_float32_matmul_precision('medium') 


def setup_data_handlers(config):
    dwi, mask, bvec, bval = load_data(config)
    
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

    # Set default values for normalization if not specified in config
    normalize_0_1 = config.get('normalize_0_1', False)
    normalize_in_out = config.get('normalize_in_out', True) #backwards compatibility
    print(f"normalize_0_1: {normalize_0_1}, normalize_in_out: {normalize_in_out}")

    train_dataset = PreBatchedDWIReprDataset(train_dwi, mask, train_bvec, train_bval, 
                                  normalize_0_1=normalize_0_1, 
                                  normalize_in_out=normalize_in_out, 
                                  bval_selection=config.get('bval_train') if config.get('bval_train') else 'all',
                                  preload_to_gpu=config.get('preload_to_gpu', False),
                                  device='cuda:0',
                                  train_batch_size=config['train_batch_size'],
                                  )
    
    no_trainworkers = 0 if config.get('preload_to_gpu', False) else 6

    train_dataloaders = DataLoader(train_dataset, 
                                   # batch_size=config['train_batch_size'], shuffle=False, num_workers=6)
                                   batch_size=None, shuffle=True, num_workers=0)
    
    # Define eval_gradients (indices of gradient directions to evaluate during training)
    # Default to evaluating low, medium, and high b-values if available
    if 'eval_gradients' in config:
        eval_gradients = config['eval_gradients']
    else:
        # Try to automatically select a few representative gradient directions
        # Check for both 'bval' and 'bvals' attributes to be more robust
        if hasattr(train_dataset, 'bvals'):
            b_values = train_dataset.bvals
        elif hasattr(train_dataset, 'bval'):
            b_values = train_dataset.bval
        else:
            # Fallback to using all gradients if no b-values found
            b_values = torch.arange(train_dataset.data.shape[-1])
            print("Warning: No b-values found in dataset. Using all gradients.")
            

        # Get indices for low, medium, and high b-values
        sorted_indices = torch.argsort(b_values)
        eval_gradients = [
            sorted_indices[0].item(),  # Lowest b-value
            sorted_indices[len(sorted_indices)//2].item(),  # Middle b-value
            sorted_indices[-1].item()  # Highest b-value
        ]
     
    if not config.get('val_skip', False):
        val_dataset = DWIReprDataset(val_dwi, mask, val_bvec, val_bval, 
                                      normalize_0_1=normalize_0_1, 
                                      normalize_in_out=normalize_in_out, 
                                      bval_selection=config.get('bval_val') if config.get('bval_val') else 'all'
                                    )
        val_slice_dataset = ValidationSliceDataset(val_dataset, 
                                             axis=config['val_slice_axis'], 
                                             offset=config['val_slice_offset'],
                                             b_indices=config.get('val_b_idx', [0, 15, 30]),
                                             normalize_0_1=config.get('normalize_0_1', True), 
                                             normalize_in_out=normalize_in_out)

        # currently same as train loader - TODO some of these stuff belongs into config
        val_volume_dataset = DWIReprDataset(val_dwi, mask, val_bvec, val_bval, 
                                        normalize_0_1=config.get('normalize_0_1', True), 
                                        normalize_in_out=config.get('normalize_0_1', True),
                                        subsampl_step=4,
                                        bval_selection=[0, 3000])
        val_slice_loader = DataLoader(val_slice_dataset, 
                                      batch_size=config['val_batch_size'], shuffle=False, num_workers=6)
        val_volume_loader = DataLoader(val_volume_dataset, 
                                      batch_size=val_volume_dataset.num_samples, shuffle=False, num_workers=6)
        val_dataloaders = {'slice': val_slice_loader, 'volume': val_volume_loader} 

        return val_slice_dataset, train_dataloaders, val_dataloaders, eval_gradients
    
    else:
        return None, train_dataloaders, None, eval_gradients



def get_checkpoint_callback(config):
    checkpoint_dirpath = config.get('checkpoint_dirpath', 'checkpoints')
    checkpoint_filename = config.get('checkpoint_filename', 'inr-{step}')
    checkpoint_save_top_k = config.get('checkpoint_save_top_k', -1)  # -1 means keep all checkpoints
    checkpoint_every_n_train_steps = config.get('checkpoint_every_n_train_steps', 5000)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename=checkpoint_filename,
        save_top_k=checkpoint_save_top_k,
        every_n_train_steps=checkpoint_every_n_train_steps,
        save_last=True,     # for now hardcoded
    )
    
    return checkpoint_callback


def main():
    args = parse_args(evaluation_mode=False)
    config = parse_config(args.config, subject=args.subject, rootdir=args.rootdir)

    val_slice_dataset, train_dataloaders, val_dataloaders, eval_gradients = setup_data_handlers(config)
    
    # setup model architecture
    inr = model_mapping[config['network_name']](config['in_size'], 
                                                config['out_size'], 
                                                hidden_size=config['hidden_size'], 
                                                num_layers=config['num_layers'])

    # setup input positional encoder
    use_pos_encoder = config.get('pos_encoder', True) #backwards compatibility
    
    if use_pos_encoder:
        print("Initializing pos encoder")
        pos_encoder = [BvecAngleEncoder(7, (4, 5, 6)),
                       SphericalPosEncoder(l_max=8, 
                                           coord_size=4, 
                                           freq_num=128,  # => 256 freq. channels in total
                                           freq_scale=1.
                                           )]
    else:
        pos_encoder = None

    # run training
    logger = TensorBoardLogger("tb_logs", name="my_model", log_graph=False, default_hp_metric=False)

    # Get the maximum b-value from the dataset for proper unnormalization
    data_bval_max = train_dataloaders.dataset.bvals.max()
    max_bval = config.get('max_bval', data_bval_max) if config.get('max_bval') else data_bval_max
    
    # Get steps_per_viz from config, default to 2000 if not specified
    steps_per_viz = config.get('steps_per_viz', 2000)
    if not os.path.exists(config["checkpoint_dirpath"]):
        os.makedirs(config["checkpoint_dirpath"])
        
    inr_lighning_kwargs={
        'network': inr,
        'train_loader': train_dataloaders,
        'val_loaders_dict': val_dataloaders,
        'lr': config['lr'], 
        'log_coords_raw': config['log_coords_raw'],
        'log_coords_encoded': config['log_coords_encoded'],
        'eval_gradients': eval_gradients,
        'name': config.get('model_name', ''),
        'max_bval': max_bval,
        'steps_per_viz': steps_per_viz,
        'export_dwi': config["val_export_dwi"],
        'epochs': config['epochs'],
        'use_cos_lrsheduler': config.get('use_cos_lrsheduler', False),
        'train_log_interval': config.get('train_log_interval', 1),
        'weight_decay': config.get('weight_decay', 0.0),
        'use_dropout': config.get('use_dropout', False)
    }

    if not config.get('val_skip', False):
        inr_lighning_kwargs['val_slice_func'] = val_slice_dataset._fold_slice,

    # Create model or load from checkpoint
    if 'checkpoint_path' in config and config['checkpoint_path']:
        print(f"Loading model from checkpoint: {config['checkpoint_path']}")

        # Load model from fixed checkpoint - don't pass pos_encoder to use the one from checkpoint
        pl_module = INRLightningModule.load_from_checkpoint(
            config['checkpoint_path'],
            **inr_lighning_kwargs
        )
    else:
        pl_module = INRLightningModule(
            pos_encoder=pos_encoder,  # Only pass pos_encoder for new models
            **inr_lighning_kwargs
            )

    # Setup checkpoint callback
    checkpoint_callback = get_checkpoint_callback(config)

    trainer_args = {
        'precision': '16-mixed',    
        'max_epochs': config['epochs'],
        'logger': logger,
        'num_sanity_val_steps': config['val_sanity_steps'],
        'accelerator': 'gpu',
        'devices': torch.cuda.device_count(),                # 4 GPUs on the current node
        'strategy': 'ddp',
        'callbacks': [checkpoint_callback],
        'profiler': 'simple'
    }


    if config['val_mode'] == 'step':
        trainer_args['val_check_interval'] = config['val_check_interval']
    elif config['val_mode'] == 'epoch':
        trainer_args['check_val_every_n_epoch'] = config['val_check_interval']
    else:
        raise ValueError("Invalid validation mode. Choose 'step' or 'epoch'.")

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(pl_module)

if __name__ == "__main__":
    main()
