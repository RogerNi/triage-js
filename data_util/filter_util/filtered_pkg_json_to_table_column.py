import json

# Replace this with the actual path to your JSON file
input_path = "filtered_pkg.json"

with open(input_path, 'r') as f:
    data = json.load(f)

for name, status in data:
    print(status)