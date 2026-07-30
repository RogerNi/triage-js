# This script is for filtering packages in the dataset based on the following criteria:
# 1. No valid sinks
# 2. Sinks not in packages (but in drivers)
# 3. Sink locations are wrong

# For valid packages, fix the package name and version if they are wrong 
# (likely due to NodeMedic parsing issues: _ or - is used for splitting where it exists in some package names)

import os
import tarfile
import json
import csv
from transformers import AutoTokenizer
from jsmin import jsmin
import re
from tqdm import tqdm


DATASET_JSON = "../new_dataset_incl_manual_checks/new_data.json"
PACKAGES_SRC_DIR = "../NPM_PKG/packages"

SRC_PATH_SUBSTITUTION = {"/nodetaint/packageData": "tmp/localpackagedata"}

exception_package = set()


def extract_js_function_from_range(text, start_line, end_line):
    lines = text.splitlines()
    start_line -= 1  # Convert to 0-based index
    end_line -= 1    # Convert to 0-based index
    if start_line < 0 or end_line >= len(lines) or start_line > end_line:
        return False
    
    # define all possible code injection or command execution sinks
    all_possible_sinks = [
        # JS code execution
        "eval",
        "Function",
        "setTimeout",
        "setInterval",
        "runInThisContext",
        "runInNewContext",
        "compileFunction",
        "Script",

        # OS command execution
        "exec",
        "execSync",
        "spawn",
        "spawnSync",
        "fork",
        "execFile",
        "execFileSync",

        # Possibly dangerous process interaction
        "env",
        "binding",
        
        # Native add-ons and dynamic loading
        "dlopen",

        # File access (indirectly enabling command execution)
        "readFile",
        "readFileSync",
        "writeFile",
        "writeFileSync",
        "appendFile",
        "appendFileSync",
        "createWriteStream"
    ]

    for i in range(start_line, end_line + 1):
        line = lines[i]
        for sink in all_possible_sinks:
            if sink in line:
                return True

    print(f"Warning: No valid sink found in the range {start_line + 1}-{end_line + 1} in the provided code: \n{lines[start_line:end_line + 1]}\nFull code: \n{text}\n{'-'*50}")
    return False

    
def list_and_read_js_files(tar_gz_path):
    js_files_content = {}
    
    # Open the tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # Iterate through all members of the tar archive
        for member in tar.getmembers():
            # Check if the member is a .js file
            if member.name.endswith('.js') and member.isfile() and "example" not in member.name:
                try:
                    # Extract the file content as a string
                    file_content = tar.extractfile(member).read().decode('utf-8')
                except Exception as e:
                    continue
                # Store the content with the file name
                js_files_content[member.name] = file_content
    
    return js_files_content

def get_lines(pkg_name, version, file_path):
    try:
        tar_path = find_src_tar(pkg_name, version)
        with tarfile.open(tar_path, 'r') as tar:
            package_name = tar_path.split(os.sep)[-2]
            new_path = substitute_path(SRC_PATH_SUBSTITUTION, file_path, package_name)
            if new_path in tar.getnames():
                file = tar.extractfile(new_path)
                if file:
                    try:
                        content = file.read().decode('utf-8')
                        return content
                    except Exception as e:
                        print(f"Error reading {new_path} in {tar_path}: {e}")
                        return ""
            else:
                # invalid package, return empty string
                return ""
                
    except Exception as e:
        print(f"Error processing {pkg_name}: {e}")
        # print traceback
        import traceback
        traceback.print_exc()
        exception_package.add((pkg_name, version))
        return ""
    

def find_src_tar(pkg_name, version):
    target_pkg_prefix = f"{pkg_name.replace('/', '%')}_{version}"
    target_pkg_path = os.path.join(PACKAGES_SRC_DIR, target_pkg_prefix)
    
    # Check if exact folder exists
    if os.path.isdir(target_pkg_path):
        target_tar_path = os.path.join(target_pkg_path, "package.tar.gz")
        if os.path.isfile(target_tar_path):
            return target_tar_path
        else:
            raise FileNotFoundError(f"Cannot find package.tar.gz in {target_pkg_path}")
    else:
        # Search for folders starting with {pkg_name}_{version}
        matching_folders = [
            folder for folder in os.listdir(PACKAGES_SRC_DIR)
            if os.path.isdir(os.path.join(PACKAGES_SRC_DIR, folder)) and folder.startswith(target_pkg_prefix)
        ]
        
        if matching_folders:
            # Use the first matching folder
            target_pkg_path = os.path.join(PACKAGES_SRC_DIR, matching_folders[0])
            target_tar_path = os.path.join(target_pkg_path, "package.tar.gz")
            if os.path.isfile(target_tar_path):
                return target_tar_path
            else:
                raise FileNotFoundError(f"Cannot find package.tar.gz in {target_pkg_path}")
        else:
            raise FileNotFoundError(f"Cannot find any folder starting with {target_pkg_prefix} in {PACKAGES_SRC_DIR}")
        
def correct_package_name_and_version(pkg_name, version):
    """In case the package name or version was parsed incorrectly, this function will correct it.
    Get the correct package name and version from folder. Refer to find src_tar function.

    Args:
        pkg_name (str): The package name to correct.
        version (str): The version to correct.
    """
    target_pkg_prefix = f"{pkg_name.replace('/', '%')}_{version}"
    target_pkg_path = os.path.join(PACKAGES_SRC_DIR, target_pkg_prefix)
    
    if os.path.isdir(target_pkg_path):
        # If the exact folder exists, return the original name and version
        return pkg_name, version
    else:
        # Search for folders starting with {pkg_name}_{version}
        matching_folders = [
            folder for folder in os.listdir(PACKAGES_SRC_DIR)
            if os.path.isdir(os.path.join(PACKAGES_SRC_DIR, folder)) and folder.startswith(target_pkg_prefix)
        ]
        
        if matching_folders:
            # Use the first matching folder to extract the correct package name and version
            correct_folder = matching_folders[0]
            # Split by last underscore to get package name and version (the last part is the version, the rest is the package name)
            parts = correct_folder.split('_')
            correct_version = parts[-1]
            correct_pkg_name = '_'.join(parts[:-1]).replace('%', '/')
            return correct_pkg_name, correct_version
        else:
            raise ValueError(f"Cannot find any folder starting with {target_pkg_prefix} in {PACKAGES_SRC_DIR}")
    
    
    
    
def substitute_path(substitution_dict, path, middle_dir=None):
    if path.startswith("eval("):
        path = path[5:]
    for old_part, new_part in substitution_dict.items():
        # If the old part is found in the path, replace it with the new part
        if old_part in path:
            path = path.replace(old_part, new_part if middle_dir is None else os.path.join(new_part, middle_dir))
            break
        
    return path


if __name__ == "__main__":
    with open(DATASET_JSON, 'r') as f:
        entries = json.load(f)["flows"]
        
    graph_metadata = []
    filtered_entries = []
    
    new_graph = []
        
    for idx, entry in tqdm(enumerate(entries)):
        written = False
        for node_idx, node_info in entry['provenance_tree'].items():
            if node_info['sink_type'] != "":
                try:
                    if node_info['file'] == "UNKNOWN" and  int(node_idx) != len(entry['provenance_tree'].items()):
                            continue
                    flows_from = ','.join(node_info.get('flows_from', []))  # Join flows_from list into a comma-separated string
                    codes = get_lines(entry['id'], entry['version'], node_info['file'] if node_info['sink_type'] != "" else "UNKNOWN")
                    if codes == "":
                        filtered_entries.append((entry['id'], "file_not_found"))
                    else:
                        if extract_js_function_from_range(codes, node_info['startLineNumber'], node_info['endLineNumber']) == True:
                            # Correct package name and version if necessary
                            pkg_name, version = correct_package_name_and_version(entry['id'], entry['version'])
                            new_graph.append({
                                "id": pkg_name,
                                "version": version,
                                "confirmed": entry['confirmed'],
                                "vuln_type": entry['vuln_type'],
                                "provenance_tree": entry['provenance_tree'],
                                "exploitable": entry['exploitable'],
                            })
                            filtered_entries.append((entry['id'], "valid"))
                        else:
                            print(f"Package: {entry['id']}, Sink Type: {node_info['sink_type']}, Operation: {node_info['operation']}, File: {node_info['file']}, Start Line: {node_info['startLineNumber']}, End Line: {node_info['endLineNumber']}")
                            print(f"{'='*40}")
                            filtered_entries.append((entry['id'], "line_number_mismatch"))
                except Exception as e:
                    print(f"Idx: {node_idx}, Info: {node_info}, E: {e}")
                written = True
            else:
                pass # not valid
            break
        if not written:
            print(f"Warning: {entry['id']} has no sink node.")
            filtered_entries.append((entry['id'], "no_sink"))
    
    print(exception_package)
    # Write the filtered entries to a new JSON file
    with open('filtered_pkg.json', 'w') as f:
        json.dump(filtered_entries, f, indent=4)

    with open('filtered_flow.json', 'w') as f:
        json.dump({"flows": new_graph}, f, indent=4)