import pandas as pd
import argparse
import os
import numpy as np

def parse_overall_metrics(metrics_text):
    metrics = {}
    for line in metrics_text:
        if line.strip():
            key, value = line.split(':')
            metrics[key.strip()] = float(value.strip())
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="csv_results", help="Directory to save CSV results for plotting")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Dictionary to store metrics for each split
    split_metrics = {}
    # Dictionary to store overall metrics for table creation
    split_overall_table = {}

    # loop through all folders in the results directory
    for folder in os.listdir(args.results_dir):
        folder_path = os.path.join(args.results_dir, folder)
        print(f"Processing folder: {folder_path}")

        if os.path.isdir(folder_path):
            # Extract subject ID and split value from folder name
            subject_id = folder
            
            # Look for split folders within subject folder
            for split_folder in os.listdir(folder_path):
                split_path = os.path.join(folder_path, split_folder)
                if os.path.isdir(split_path):
                    split = split_folder.split('_')[-1]  # Get split value
                    
                    if split not in split_metrics:
                        split_metrics[split] = {
                            'overall': [],
                            'bvalue': []
                        }
                    
                    # Load the per-b-value results
                    metrics_file = os.path.join(split_path, "evaluation_metrics.csv")
                    if os.path.exists(metrics_file):
                        df = pd.read_csv(metrics_file)
                        df['subject_id'] = subject_id  # Add subject ID to track which subject this came from
                        split_metrics[split]['bvalue'].append(df)
                        print(f"Loaded bvalue metrics for subject {subject_id}, split {split}")
                    
                    # Load the overall metrics
                    overall_file = os.path.join(split_path, "overall_metrics.txt")
                    if os.path.exists(overall_file):
                        with open(overall_file, "r") as f:
                            metrics = parse_overall_metrics(f.readlines())
                            metrics['subject_id'] = subject_id  # Add subject ID to track which subject this came from
                            split_metrics[split]['overall'].append(metrics)
                            print(f"Loaded overall metrics for subject {subject_id}, split {split}")

    # Create a table of overall metrics by split
    print("\n" + "="*100)
    print("Overall Metrics by Split")
    print("="*100)
    
    # First, collect all metrics for the table
    for split in sorted(split_metrics.keys()):
        if split_metrics[split]['overall']:
            overall_df = pd.DataFrame(split_metrics[split]['overall'])
            metrics_cols = [col for col in overall_df.columns if col != 'subject_id']
            split_overall_table[split] = overall_df[metrics_cols].mean()

    # Create and display the table
    if split_overall_table:
        table_df = pd.DataFrame(split_overall_table).T
        # Format the numbers to 6 decimal places
        table_df = table_df.round(6)
        print("\nSplit-wise Overall Metrics:")
        print(table_df.to_string())
        
        # Save overall metrics table to CSV
        overall_csv_path = os.path.join(args.output_dir, "overall_metrics_by_split.csv")
        table_df.to_csv(overall_csv_path)
        print(f"\nSaved overall metrics by split to: {overall_csv_path}")

    # Process results for each split
    for split in sorted(split_metrics.keys()):
        print(f"\n{'='*70}")
        print(f"Results for Split {split}")
        print(f"{'='*70}")

        # Print number of files loaded for this split
        print(f"\nNumber of bvalue files loaded: {len(split_metrics[split]['bvalue'])}")
        print(f"Number of overall files loaded: {len(split_metrics[split]['overall'])}")

        # Calculate averages and std for overall metrics
        if split_metrics[split]['overall']:
            # Convert list of dictionaries to DataFrame for easier processing
            overall_df = pd.DataFrame(split_metrics[split]['overall'])
            metrics_cols = [col for col in overall_df.columns if col != 'subject_id']
            
            avg_overall = overall_df[metrics_cols].mean()
            std_overall = overall_df[metrics_cols].std()
            
            # Create DataFrame for overall metrics with mean and std
            overall_mean_std_df = pd.DataFrame({
                'metric': metrics_cols,
                'mean': [avg_overall[metric] for metric in metrics_cols],
                'std': [std_overall[metric] for metric in metrics_cols]
            })
            
            # Save to CSV
            overall_split_csv = os.path.join(args.output_dir, f"split_{split}_overall_metrics.csv")
            overall_mean_std_df.to_csv(overall_split_csv, index=False)
            print(f"Saved overall metrics for split {split} to: {overall_split_csv}")
            
            print("\nAverage Overall Metrics (mean ± std):")
            for metric in metrics_cols:
                print(f"{metric}: {avg_overall[metric]:.6f} ± {std_overall[metric]:.6f}")

        # Calculate averages and std for per-b-value metrics
        if split_metrics[split]['bvalue']:
            combined_df = pd.concat(split_metrics[split]['bvalue'])
            avg_by_bvalue = combined_df.groupby('b_value').mean(numeric_only=True)
            std_by_bvalue = combined_df.groupby('b_value').std(numeric_only=True)
            
            # Create DataFrame for b-value metrics with mean and std
            bvalue_results = []
            for b_value in avg_by_bvalue.index:
                for metric in avg_by_bvalue.columns:
                    if metric != 'subject_id':
                        bvalue_results.append({
                            'b_value': b_value,
                            'metric': metric,
                            'mean': avg_by_bvalue.loc[b_value, metric],
                            'std': std_by_bvalue.loc[b_value, metric]
                        })
            
            bvalue_results_df = pd.DataFrame(bvalue_results)
            
            # Save to CSV
            bvalue_csv = os.path.join(args.output_dir, f"split_{split}_bvalue_metrics.csv")
            bvalue_results_df.to_csv(bvalue_csv, index=False)
            print(f"Saved b-value metrics for split {split} to: {bvalue_csv}")
            
            print("\nAverage Metrics by B-value (mean ± std):")
            for b_value in avg_by_bvalue.index:
                print(f"\nB-value: {b_value}")
                for metric in avg_by_bvalue.columns:
                    if metric != 'subject_id':  # Skip the subject_id column
                        mean_val = avg_by_bvalue.loc[b_value, metric]
                        std_val = std_by_bvalue.loc[b_value, metric]
                        print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")

if __name__ == "__main__":
    main()

    