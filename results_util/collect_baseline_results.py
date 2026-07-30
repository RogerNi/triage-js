import re
import json
import statistics
from collections import defaultdict
from tabulate import tabulate

def parse_metrics_from_file(filepath):
    pattern = re.compile(r"(?P<model>.*?) on Test - (?P<metrics>{.*})")
    results = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                model = match.group("model").strip()
                metrics_str = match.group("metrics")
                try:
                    metrics = json.loads(metrics_str.replace("'", '"'))
                    results[model].append(metrics)
                except json.JSONDecodeError:
                    print(f"[WARN] Failed to parse metrics in: {filepath}")
    return results

def aggregate_results(file_list):
    aggregated = defaultdict(list)

    for file_path in file_list:
        model_metrics = parse_metrics_from_file(file_path)
        for model, metrics_list in model_metrics.items():
            aggregated[model].extend(metrics_list)

    summary = {}
    for model, metrics_list in aggregated.items():
        summary[model] = {}
        keys = metrics_list[0].keys()
        for k in keys:
            values = [m[k] for m in metrics_list]
            mean = statistics.mean(values)
            var = statistics.variance(values) if len(values) > 1 else 0.0
            summary[model][k] = (mean, var)
    return summary


def format_table(summary):
    # Define the desired metric order explicitly
    desired_order = ["f1", "precision", "recall", "accuracy"]
    headers = ["Model"] + [metric.capitalize() for metric in desired_order]
    rows = []

    for model in sorted(summary.keys()):
        row = [model]
        for metric in desired_order:
            if metric in summary[model]:
                mean, var = summary[model][metric]
                row.append(f"{mean:.4f} (±{var:.2e})")
            else:
                row.append("-")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="github")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=str, required=True,
                        help="List of result files, e.g., file1.txt file2.txt")
    args = parser.parse_args()

    summary = aggregate_results(args.files)
    table_str = format_table(summary)
    print(table_str)