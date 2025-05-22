from networks.inrs import SirenINR, WireINR
from networks.leaky_relu_inr import LeakyReLUINR
import argparse
import yaml


model_mapping = {
    "leaky_relu_mlp": LeakyReLUINR,
    "wiren_mlp": WireINR,
    "siren_mlp": SirenINR,
}

def parse_args(evaluation_mode=False):
    parser = argparse.ArgumentParser(
        description='Train or evaluate an INR model for DWI representation'
    )
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    if evaluation_mode:
        parser.add_argument('output_dir', help='Directory to save the evaluation results')
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID to process. Will replace "subject" placeholders in config paths.')
    parser.add_argument('--no_save_volumes', action='store_true',
                        help='Do not save the predicted and ground truth volumes as nifti files (evaluation only)')
    parser.add_argument('--no_denorm', action='store_true',
                        help='Do not denormalize the volumes (evaluation only)')
    parser.add_argument('--rootdir', type=str, default=None,
                        help='Root data directory, will replace all occurances of <ROOT> in config.')
    return parser.parse_args()

def parse_config(config_path, subject=None, rootdir=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for key in config.keys():
        if isinstance(config[key], str) and subject is not None and "subject" in config[key]:
            config[key] = config[key].replace("subject", subject)
        if isinstance(config[key], str) and rootdir is not None and "<ROOT>" in config[key]:
            config[key] = config[key].replace("<ROOT>", rootdir)
    return config