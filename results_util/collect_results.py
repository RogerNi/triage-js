import os
import re
import argparse
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description="Process log files within a time range to extract evaluation metrics.")
parser.add_argument("log_dir", type=str, help="Path to the directory containing log files")
parser.add_argument("start_time", type=str, help="Start time in format YYYYMMDD_HHMMSS")
parser.add_argument("end_time", type=str, help="End time in format YYYYMMDD_HHMMSS")

args = parser.parse_args()

# Convert start and end times to datetime objects
start_time = datetime.strptime(args.start_time, "%Y%m%d_%H%M%S")
end_time = datetime.strptime(args.end_time, "%Y%m%d_%H%M%S")

# Regular expression to match the "Test Metrics" line
metrics_pattern = re.compile(r"Test Metrics: (.*)")

# Function to extract run name from the filename
def extract_run_name(filename):
    return filename.replace(".log", "")

# Process log files
results = []
for filename in os.listdir(args.log_dir):
    if filename.endswith(".log"):
        # Extract timestamp from the filename
        match = re.search(r"train_(\d{8}_\d{6})_", filename)
        if match:
            log_time = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")

            # Check if the log file is within the time range
            if start_time <= log_time <= end_time:
                log_path = os.path.join(args.log_dir, filename)
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Search for the last occurrence of "Test Metrics"
                for line in reversed(lines):
                    match = metrics_pattern.search(line)
                    if match:
                        try:
                            metrics = eval(match.group(1))  # Convert string to dict safely
                            f1 = metrics.get("eval_f1", "N/A")
                            precision = metrics.get("eval_precision", "N/A")
                            recall = metrics.get("eval_recall", "N/A")
                            accuracy = metrics.get("eval_accuracy", "N/A")

                            run_name = extract_run_name(filename)
                            results.append(f"{run_name}\t{f1},{precision},{recall},{accuracy}")
                        except Exception as e:
                            print(f"Error parsing metrics in {filename}: {e}")
                        break  # Stop searching after the first match from the bottom

# Print results
for result in results:
    print(result)
    