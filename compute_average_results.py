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
    args = parser.parse_args()

    # Initialize lists to store metrics from all subjects
    all_overall_metrics = []
    all_bvalue_metrics = []

    # loop through all folders in the results directory
    for folder in os.listdir(args.results_dir):
        folder_path = os.path.join(args.results_dir, folder)
        if os.path.isdir(folder_path):
            # Load the per-b-value results
            metrics_file = os.path.join(folder_path, "evaluation_metrics.csv")
            if os.path.exists(metrics_file):
                all_bvalue_metrics.append(pd.read_csv(metrics_file))
            
            # Load the overall metrics
            overall_file = os.path.join(folder_path, "overall_metrics.txt")
            if os.path.exists(overall_file):
                with open(overall_file, "r") as f:
                    metrics = parse_overall_metrics(f.readlines())
                    all_overall_metrics.append(metrics)

    # Calculate averages and std for overall metrics
    if all_overall_metrics:
        avg_overall = {}
        std_overall = {}
        for metric in all_overall_metrics[0].keys():
            values = [m[metric] for m in all_overall_metrics]
            avg_overall[metric] = np.mean(values)
            std_overall[metric] = np.std(values)
        
        print("\nAverage Overall Metrics (mean ± std):")
        for metric in avg_overall.keys():
            print(f"{metric}: {avg_overall[metric]:.6f} ± {std_overall[metric]:.6f}")

    # Calculate averages and std for per-b-value metrics
    if all_bvalue_metrics:
        combined_df = pd.concat(all_bvalue_metrics)
        avg_by_bvalue = combined_df.groupby('b_value').mean()
        std_by_bvalue = combined_df.groupby('b_value').std()
        
        print("\nAverage Metrics by B-value (mean ± std):")
        for b_value in avg_by_bvalue.index:
            print(f"\nB-value: {b_value}")
            for metric in avg_by_bvalue.columns:
                mean_val = avg_by_bvalue.loc[b_value, metric]
                std_val = std_by_bvalue.loc[b_value, metric]
                print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")

if __name__ == "__main__":
    main()

    