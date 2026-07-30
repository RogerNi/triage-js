import os
import re
import ast
import statistics
from collections import defaultdict
from tabulate import tabulate


def parse_test_metrics(filepath):
    """
    Extract test metrics from lines like:
    - metrics = {...}
    - Test Metrics: {...}
    - Baseline results: {'Random Forest': {...}, ...}
    Returns a tuple of two dicts:
      - test_metrics: {"model": {f1, precision, recall, accuracy}}
      - baseline_metrics: {baseline_name: {f1, precision, recall, accuracy}}
    """
    test_patterns = [
        re.compile(r"metrics\s*=\s*(\{.*\})\)?"),
        re.compile(r"Test Metrics:\s*(\{.*\})\)?"),
    ]
    baseline_pattern = re.compile(r"Baseline results:\s*(\{.*\})\)?")

    test_metrics = {}
    baseline_metrics = {}

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = reversed(f.readlines())

        for line in lines:
            # Try baseline first
            match = baseline_pattern.search(line)
            if match:
                try:
                    data = ast.literal_eval(match.group(1))
                    if isinstance(data, dict):
                        for model_name, model_metrics in data.items():
                            if isinstance(model_metrics, dict):
                                baseline_metrics[model_name] = {
                                    k: float(model_metrics[k])
                                    for k in ("f1", "precision", "recall", "accuracy")
                                    if k in model_metrics
                                }
                except Exception:
                    continue

            # Then try regular test metric lines
            for pattern in test_patterns:
                match = pattern.search(line)
                if match:
                    try:
                        data = ast.literal_eval(match.group(1))
                        metrics = {}
                        for k in ("f1", "precision", "recall", "accuracy"):
                            if k in data:
                                metrics[k] = float(data[k])
                            elif f"test_{k}" in data:
                                metrics[k] = float(data[f"test_{k}"])
                        if metrics:
                            test_metrics["default"] = metrics
                    except Exception:
                        continue

    return test_metrics, baseline_metrics


def collect_logs_from_flat_dir(log_dir):
    """
    Collect metrics from .log files in the given directory.
    Returns:
      - test_results: {model_name: [metrics]}
      - baseline_results: {baseline_name: [metrics]}
    """
    test_results = defaultdict(list)
    baseline_results = defaultdict(list)

    for fname in os.listdir(log_dir):
        if fname.endswith(".log"):
            filepath = os.path.join(log_dir, fname)
            base_name = os.path.splitext(fname)[0]
            base_model_name = re.sub(r"_\d{4}$", "", base_name)

            test_metrics, baselines = parse_test_metrics(filepath)

            for model, metrics in test_metrics.items():
                full_name = base_model_name if model == "default" else model
                test_results[full_name].append(metrics)

            for model, metrics in baselines.items():
                baseline_results[model].append(metrics)

    return test_results, baseline_results


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

    parser = argparse.ArgumentParser(
        description="Summarize test and baseline metrics from log files."
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        required=True,
        help="List of log directories containing .log files.",
    )
    args = parser.parse_args()

    aggregated_test = defaultdict(list)
    aggregated_baseline = defaultdict(list)

    for log_dir in args.prefixes:
        test, baseline = collect_logs_from_flat_dir(log_dir)
        for model, metrics_list in test.items():
            aggregated_test[model].extend(metrics_list)
        for model, metrics_list in baseline.items():
            aggregated_baseline[model].extend(metrics_list)

    test_summary = summarize_model_metrics(aggregated_test)
    baseline_summary = summarize_model_metrics(aggregated_baseline)

    if test_summary:
        print("### Test Metrics:")
        print(format_table(test_summary, title="Model"))

    if baseline_summary:
        print("\n### Baseline Metrics:")
        print(format_table(baseline_summary, title="Baseline"))