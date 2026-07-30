#!/bin/bash
# This script unzips the file "preprocessed.sh" (a zip archive) into a "dataset" folder.
# It works regardless of the current working directory by determining the script's location.

# Get the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the zip file and the target dataset folder.
ZIP_FILE="${SCRIPT_DIR}/preprocessed.zip"
TARGET_DIR="${SCRIPT_DIR}/../dataset"

# Check if the dataset folder exists; if not, create it.
if [ ! -d "$TARGET_DIR" ]; then
    echo "Dataset folder not found. Creating $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
fi

# Unzip the file into the dataset folder.
echo "Unzipping $ZIP_FILE into $TARGET_DIR..."
unzip "$ZIP_FILE" -d "$TARGET_DIR"