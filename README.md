# INR for Diffusion MRI


## Setup

```bash
pip install -r requirements.txt
pip install -e .
```


## Data
Add your config to `src/cfg/` specifying paths to DWI, brain mask, b-values and b-vectors.


## Training
To train a model using main.py:

```bash
python src/main.py <config_path> [--subject SUBJECT_ID]
```

Arguments:
- `config_path`: Path to the YAML configuration file
- `--subject`: (Optional) Subject ID to process. If provided the "subject" keyword in the config will be replaced with the specific subject id.

Example with one HCP subject:
```bash
python src/main.py cfg/one_hcp.yml --subject 107422
```

## Evaluation
To evaluate a trained model:

```bash
python src/evaluation.py <config_path> <output_dir> [--subject SUBJECT_ID] [--no_save_volumes] [--no_denorm]
```

Arguments:
- `config_path`: Path to the YAML configuration file
- `output_dir`: Directory to save evaluation results
- `--subject`: (Optional) Subject ID to evaluate. Replaces "subject" placeholders in config paths
- `--no_save_volumes`: (Optional) Skip saving predicted and ground truth volumes
- `--no_denorm`: (Optional) Skip denormalization of volumes 

Example:
```bash
python src/evaluation.py cfg/one_hcp.yml results/evaluation --subject 107422 --no_denorm
```

### Evaluation Process
The evaluation script performs the following steps:
1. Loads the trained model from the checkpoint specified in the config
2. Creates training and validation datasets
3. Evaluates the model on the complete volume
4. Computes various metrics for each b-value
5. Saves results and generates visualizations

### Evaluation Metrics
The evaluation generates the following metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Relative Error (%)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

### Output Files
Results are saved in the specified output directory:
- `evaluation_metrics.csv`: Raw metrics for all evaluations
- `metrics_summary_by_bvalue.csv`: Summary metrics grouped by b-value
- `metrics_by_bvalue.png`: Visualization of metrics across b-values
- `overall_metrics.txt`: Overall performance metrics
- `dwi/`: Directory containing:
  - `dwi_pred.nii.gz`: Predicted DWI volumes
  - `dwi_gt.nii.gz`: Ground truth DWI volumes
  - `bvecs.txt`: Gradient directions
  - `bvals.txt`: B-values

## Evaluation
To evaluate a trained model:
```bash
python src/evaluation.py <config_path> <output_dir> [--subject SUBJECT_ID] [--no_save_volumes] [--no_denorm]
```

Arguments:
- `config_path`: Path to the YAML configuration file
- `output_dir`: Directory to save evaluation results
- `--subject`: (Optional) Subject ID to evaluate. Replaces "subject" placeholders in config paths
- `--no_save_volumes`: (Optional) Skip saving predicted and ground truth volumes
- `--no_denorm`: (Optional) Skip denormalization of volumes

### SHORE Model Evaluation
To evaluate using SHORE model:
```bash
python src/evaluation_shore.py <config_path> <output_dir> [--subject SUBJECT_ID] [--no_save_volumes] [--no_denorm]
```

## Evaluation Metrics
The evaluation generates the following metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Relative Error (%)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Results are saved in the specified output directory:
- `evaluation_metrics.csv`: Raw metrics for all evaluations
- `metrics_summary_by_bvalue.csv`: Summary metrics grouped by b-value
- `metrics_by_bvalue.png`: Visualization of metrics across b-values
- `overall_metrics.txt`: Overall performance metrics
- `dwi/`: Directory containing predicted and ground truth volumes (if saved)

