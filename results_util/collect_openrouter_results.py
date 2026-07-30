import os
import re
import statistics
from collections import defaultdict
from tabulate import tabulate


def parse_strict_eval_metrics(filepath):
    """Extract metrics from multi-line '[Strict Evaluation]' block in a log file."""
    metrics = {}
    found_strict = False
    lines_after_strict = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if "[Strict Evaluation]" in line:
                found_strict = True
            elif found_strict:
                if line == "" or line.startswith("["):  # end of block
                    break
                lines_after_strict.append(line)

    for line in lines_after_strict:
        try:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = float(val.strip())
                if key in ["f1", "precision", "recall", "accuracy"]:
                    metrics[key] = val
        except ValueError:
            continue

    return metrics


def collect_logs_from_seed_dirs(base_prefix):
    """Collect metrics from log files under <base_prefix>_seed_<seed>/*.log"""
    base_dir = os.path.dirname(base_prefix)
    prefix_base = os.path.basename(base_prefix).split("_seed_")[0]

    seed_dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix_base + "_seed_")]

    model_results = defaultdict(list)
    for seed_dir in seed_dirs:
        full_path = os.path.join(base_dir, seed_dir)
        if os.path.isdir(full_path):
            for fname in os.listdir(full_path):
                if fname.endswith(".log"):
                    model_name = fname.replace(".log", "")
                    filepath = os.path.join(full_path, fname)
                    metrics = parse_strict_eval_metrics(filepath)
                    if metrics:
                        model_results[model_name].append(metrics)
    return model_results


def summarize_model_metrics(model_results):
    """Compute mean and variance per model (across seeds)."""
    summary = {}
    for model, runs in model_results.items():
        summary[model] = {}
        for key in runs[0].keys():
            values = [run[key] for run in runs]
            mean = statistics.mean(values)
            var = statistics.variance(values) if len(values) > 1 else 0.0
            summary[model][key] = (mean, var)
    return summary


def group_by_model_family(summary_dict):
    """Group models by family (e.g., 'model_xxx') and average means across seeds."""
    grouped = defaultdict(list)

    for model_name, metrics in summary_dict.items():
        family = re.sub(r'_\d{4}$', '', model_name)
        grouped[family].append(metrics)

    aggregated = {}
    for family, runs in grouped.items():
        aggregated[family] = {}
        for key in runs[0].keys():
            values = [run[key][0] for run in runs if isinstance(run[key], tuple)]
            if values:
                mean = statistics.mean(values)
                var = statistics.variance(values) if len(values) > 1 else 0.0
                aggregated[family][key] = (mean, var)
    return aggregated


def format_table(summary, title="Model"):
    desired_order = ["f1", "precision", "recall", "accuracy"]
    headers = [title] + [m.capitalize() for m in desired_order]
    rows = []
    for model in sorted(summary.keys()):
        row = [model]
        for m in desired_order:
            if m in summary[model]:
                mean, var = summary[model][m]
                row.append(f"{mean:.4f} (±{var:.2e})")
            else:
                row.append("-")
        rows.append(row)
    return tabulate(rows, headers=headers, tablefmt="github")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True,
                        help="Prefix path like 'logs/train_20250714_211232_zero_shot_seed_{seed}'")
    args = parser.parse_args()

    # Parse and summarize
    model_results = collect_logs_from_seed_dirs(args.prefix)
    model_summary = summarize_model_metrics(model_results)
    family_summary = group_by_model_family(model_summary)

    # Output
    print("### Individual Model Summary:")
    print(format_table(model_summary, title="Model"))
    print("\n### Model Family Summary (Averaged Across Seeds):")
    print(format_table(family_summary, title="Model Family"))