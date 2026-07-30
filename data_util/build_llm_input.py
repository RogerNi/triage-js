import os
import tarfile
import json
import csv
from transformers import AutoTokenizer
from jsmin import jsmin
import re
from tqdm import tqdm


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

MAX_TOKENS = 1024

# DATASET_JSON = "../new_dataset_incl_manual_checks/new_data.json"
# DATASET_JSON = "../new_dataset_incl_manual_checks/filtered_flows.json"
DATASET_JSON = "dataset/filtered_flows.json"
PACKAGES_SRC_DIR = "../NPM_PKG/packages"

SRC_PATH_SUBSTITUTION = {"/nodetaint/packageData": "tmp/localpackagedata"}

# output path
# NODE_INFO_CSV = 'node_information_llm.csv'
NODE_INFO_CSV = 'dataset/node_information_per_graph.csv'

exception_package = set()


def truncate_text_by_lines_whole_by_tokens(text, max_tokens, start_line, end_line, tokenizer):
    """
    Truncate text to a maximum token count while preserving specified lines as whole lines.

    Args:
        text (str): The full text (e.g., code).
        max_tokens (int): The maximum allowed number of tokens.
        start_line (int): Starting line number (1-indexed) of the portion to preserve.
        end_line (int): Ending line number (1-indexed) of the portion to preserve.
        tokenizer: A tokenizer object with `encode` and `decode` methods (e.g., from Hugging Face).

    Returns:
        str: Truncated text.
    """
    lines = text.splitlines()

    # Convert 1-based indexing to 0-based
    start_line -= 1
    end_line -= 1

    if start_line < 0 or end_line >= len(lines) or start_line > end_line:
        print(f"Invalid line numbers: {start_line + 1} - {end_line + 1}")
        start_line = max(0, min(start_line, len(lines) - 1))
        end_line = max(start_line, min(end_line, len(lines) - 1))

    center_lines = lines[start_line:end_line + 1]
    center_text = "\n".join(center_lines)
    center_tokens = tokenizer.encode(center_text, add_special_tokens=False)
    center_token_len = len(center_tokens)

    if center_token_len > max_tokens:
        # Truncate center lines directly to token limit
        truncated_tokens = center_tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    remaining_tokens = max_tokens - center_token_len

    # Accumulate head lines in reverse
    head_lines = []
    head_token_len = 0
    for line in reversed(lines[:start_line]):
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)
        if head_token_len + len(line_tokens) > remaining_tokens // 2:
            break
        head_lines.append(line)
        head_token_len += len(line_tokens)
    head_lines.reverse()

    # Accumulate tail lines
    tail_lines = []
    tail_token_len = 0
    for line in lines[end_line + 1:]:
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)
        if tail_token_len + len(line_tokens) > remaining_tokens // 2:
            break
        tail_lines.append(line)
        tail_token_len += len(line_tokens)

    # Combine parts
    truncated = []
    if head_lines:
        truncated.extend(head_lines)
    truncated.extend(center_lines)
    if tail_lines:
        truncated.extend(tail_lines)

    return "\n".join(truncated).strip()


def extract_js_function_from_range(text, start_line, end_line):
    """
    Extract the outermost JavaScript function that contains the center line.

    Args:
        text (str): Full JavaScript source code.
        start_line (int): Start line number (1-indexed).
        end_line (int): End line number (1-indexed).

    Returns:
        str: The function block containing the center line, or an empty string.
    """
    lines = text.splitlines()
    num_lines = len(lines)

    # Compute center line (1-based → 0-based)
    center_line = (start_line + end_line) // 2
    center_idx = center_line - 1

    if center_idx < 0 or center_idx >= num_lines:
        return ""

    # Step 1: Search upward for function start (heuristically)
    function_start = None
    brace_level = 0
    function_keywords = re.compile(r"""
        ^\s*                         # optional indentation
        (
            (function\s+\w+\s*\([^)]*\))     | # function foo(...) {
            (\w+\s*=\s*function\s*\([^)]*\)) | # const foo = function(...) {
            (\w+\s*=\s*\([^)]*\)\s*=>)       | # const foo = (...) => {
            (\w+\s*:\s*function\s*\([^)]*\))   # object method: foo: function(...) {
        )
        """, re.VERBOSE)

    for i in range(center_idx, -1, -1):
        line = lines[i]
        if "{" in line and function_keywords.search(line):
            function_start = i
            break

    if function_start is None:
        return ""

    # Step 2: Search downward for the matching closing brace
    function_end = None
    for i in range(function_start, num_lines):
        brace_level += lines[i].count("{")
        brace_level -= lines[i].count("}")
        if brace_level == 0:
            function_end = i
            break

    if function_end is None:
        return ""

    return "\n".join(lines[function_start:function_end + 1]).strip()
    
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
                # first try using "main" in package.json
                package_name_no_version = "_".join(package_name.split("_")[:-1]).replace("%", "/")
                package_json = tar.extractfile(os.path.join("tmp/localpackagedata", package_name, package_name_no_version, "node_modules", package_name_no_version, "package.json"))
                package_data = json.loads(package_json.read().decode('utf-8'))
                main_file = package_data.get("main")
                if main_file:
                    new_path = os.path.join("tmp/localpackagedata", package_name, package_name_no_version, "node_modules", package_name_no_version, os.path.relpath(main_file))
                    file = tar.extractfile(new_path)
                    if file is None:
                        new_path = os.path.join(new_path, "index.js")
                        file = tar.extractfile(new_path)
                    content = file.read().decode('utf-8')
                    return content

                else:          
                    js_files = list_and_read_js_files(tar_path)
                    if len(js_files) == 0:
                        raise FileNotFoundError(f"No .js files found in {tar_path}")
                    if len(js_files) > 1:
                        print(f"Multiple .js files found in {tar_path}. Combining all files.")
                return "\n".join(js_files.values())
                
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

    # Writing the second CSV file (detailed node information with graph_idx)
    with open(NODE_INFO_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['graph_idx', 'node_idx', 'package', 'version', 'operation', 'value', 'file', 'code_full', 'code', 
                        #  'code_func', 
                         'startLineNumber', 'startColumnNumber', 'endLineNumber', 'endColumnNumber', 'tainted', 'flows_from', 'sink_type'])
        
        for idx, entry in tqdm(enumerate(entries)):
            written = False
            for node_idx, node_info in entry['provenance_tree'].items():
                if node_info['sink_type'] != "" or int(node_idx) == len(entry['provenance_tree'].items()):
                    try:
                        if node_info['file'] == "UNKNOWN" and  int(node_idx) != len(entry['provenance_tree'].items()):
                                continue
                        flows_from = ','.join(node_info.get('flows_from', []))  # Join flows_from list into a comma-separated string
                        codes = get_lines(entry['id'], entry['version'], node_info['file'] if node_info['sink_type'] != "" else "UNKNOWN")
                        codes_full = codes
                        if len(tokenizer.tokenize(codes)) > MAX_TOKENS:
                            # print(f"Warning: {entry['id']} has too many tokens: {len(tokenizer.tokenize(codes))}, use jsmin...")
                            # codes_jsmin = jsmin(codes)
                            # if len(tokenizer.tokenize(codes_jsmin)) > MAX_TOKENS:
                            #     print(f"\tWarning: {entry['id']} still has too many tokens: {len(tokenizer.tokenize(codes_jsmin))}, truncating...")
                            if "/nodetaint/packageData" not in node_info["file"]:
                                node_info['startLineNumber'] = -1
                                node_info['endLineNumber'] = -1
                            codes = truncate_text_by_lines_whole_by_tokens(codes, MAX_TOKENS, node_info['startLineNumber'], node_info['endLineNumber'], tokenizer=tokenizer)
                            # else:
                            #     codes = codes_jsmin
                        # codes_func = extract_js_function_from_range(codes_full, node_info['startLineNumber'], node_info['endLineNumber'])
                        writer.writerow([
                            idx,  # Add graph_idx to the CSV (use graph index instead of graph_id)
                            node_idx,
                            entry['id'],
                            entry['version'],
                            node_info['operation'],
                            json.dumps(node_info['value'] if 'value' in node_info else None),  # Nodes with 'value' are filled with 'null'
                            node_info['file'],
                            codes_full,
                            codes,
                            # codes_func,
                            node_info['startLineNumber'],
                            node_info['startColumnNumber'],
                            node_info['endLineNumber'],
                            node_info['endColumnNumber'],
                            node_info['tainted'],
                            flows_from,
                            node_info['sink_type']
                        ])
                    except Exception as e:
                        print(f"Idx: {node_idx}, Info: {node_info}, E: {e}")
                    written = True
                    break
            if not written:
                print(f"Warning: {entry['id']} has no sink node.")
    
    print(exception_package)
                
