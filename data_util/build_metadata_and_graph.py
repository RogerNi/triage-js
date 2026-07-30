import csv
import json
import torch
import pandas as pd 
from argparse import ArgumentParser
from torch_geometric.data import Data

# example usage: python data_util/build_metadata_and_graph.py \
#                   --dataset dataset/raw.json \
#                   --graph-out dataset/provenance_graphs.pt \
#                   --matrix-out dataset/graph_features_matrix.csv \
#                   --basic-meta-csv dataset/basic_metadata.csv

# Define the features to extract
allfeats = [
    "ops", "sink_t", "vult_t", "tainted"
]


# ops_mapping = {'Untainted': 0, '+': 1, 'object.Unary': 2, 'call:exec': 3, 'object.GetField': 4, 'precise:string.concat': 5, 'Tainted': 6, 'call:__jalangi_set_taint__': 7, 'call:concat': 8, 'call:charAt': 9, 'call:push': 10, 'call:hasOwnProperty': 11, 'call:assignValue': 12, 'call:replace': 13, 'call:gravar': 14, 'call:test': 15, 'call:SynTree': 16, 'call:forToken': 17, 'call:matchAll': 18, 'string.GetField': 19, 'call:Anonymous Function': 20, 'call:spawnWithSignal': 21, 'precise:string.replace': 22, 'call:f': 23, 'call:formatar': 24, 'call:parse': 25, 'call:run': 26, 'imprecise:exec': 27, 'call:parseInternal': 28, 'call:execFn': 29, 'precise:string.charAt': 30, 'call:compile': 31, 'call:keys': 32, 'model:string.split': 33, 'model:array.join': 34, 'imprecise:keys': 35, 'call:arrayPush': 36, 'imprecise:concat': 37, 'call:Function': 38, 'call:eval': 39, 'call:_typeof': 40, 'call:assign': 41, 'call:execSync': 42, 'call:bind': 43, 'call:isArray': 44, 'call:Parser': 45, 'call:slice': 46, 'call:spawn': 47, 'call:stringify': 48, 'call:Webcam': 49, '-': 50, 'call:merge': 51, 'call:calculate': 52, 'call:parseMathLib': 53, 'call:parseParentheses': 54, 'call:substr': 55, 'call:defineProperty': 56, 'call:isConstructorOrProto': 57, 'call:setKey': 58, 'imprecise:stringify': 59, 'precise:string.slice': 60, 'call:_matchText': 61, 'model:array.map': 62, 'call:indexOf': 63, 'call:render': 64, 'call:spawnSync': 65, 'call:String': 66, 'imprecise:slice': 67, 'call:_classCallCheck': 68, 'call:addOption': 69, 'imprecise:assign': 70, 'call:command': 71, 'call:execute': 72, 'call:toString': 73, 'call:parseNonShell': 74, 'call:log': 75, 'call:compileQuickSelect': 76, 'call:lookupCache': 77, 'call:permuteOrder': 78, 'call:process': 79, 'call:substring': 80, 'call:STRIDE': 81, 'call:_expectMultiOp': 82, 'call:_expectCalc': 83, 'call:_expectTertiary': 84, 'call:_expectExpression': 85, 'call:contains': 86, 'call:setDefaults': 87, 'call:match': 88, 'imprecise:match': 89, 'call:generate_format_string': 90, 'call:_expectUnary': 91, 'call:_expectMemberOrCall': 92, 'call:existsSync': 93, 'call:flags': 94, 'call:io': 95, 'call:template': 96, 'imprecise:toString': 97, 'call:symbolObservablePonyfill': 98, 'call:Lexer': 99}
ops_mapping = {'Untainted': 0, 'call:exec': 1, '+': 2, 'object.Unary': 3, 'precise:string.concat': 4, 'object.GetField': 5, 'call:concat': 6, 'Tainted': 7, 'call:__jalangi_set_taint__': 8, 'call:replace': 9, 'call:charAt': 10, 'call:test': 11, 'call:spawnWithSignal': 12, 'call:push': 13, 'call:Anonymous Function': 14, 'precise:string.replace': 15, 'call:matchAll': 16, 'call:parse': 17, 'call:run': 18, 'call:compile': 19, 'imprecise:exec': 20, 'call:keys': 21, 'call:arrayPush': 22, 'call:parseInternal': 23, 'call:execFn': 24, 'model:array.join': 25, 'imprecise:keys': 26, 'precise:string.charAt': 27, 'model:string.split': 28, 'call:Function': 29, 'string.GetField': 30, 'imprecise:concat': 31, 'call:bind': 32, 'call:execSync': 33, 'call:assign': 34, 'call:Parser': 35, 'call:spawn': 36, 'call:slice': 37, 'call:eval': 38, 'call:isArray': 39, 'call:stringify': 40, '-': 41, 'call:indexOf': 42, 'call:merge': 43, 'call:defineProperty': 44, 'imprecise:stringify': 45, 'imprecise:slice': 46, 'precise:string.slice': 47, 'call:String': 48, 'call:substr': 49, 'call:addOption': 50, 'call:compileQuickSelect': 51, 'call:lookupCache': 52, 'call:permuteOrder': 53, 'call:spawnSync': 54, 'model:array.map': 55, 'call:STRIDE': 56, 'call:contains': 57, 'call:log': 58, 'call:_typeof': 59, 'call:execute': 60, 'call:substring': 61, 'call:parseNonShell': 62, 'call:generate_format_string': 63, 'call:_classCallCheck': 64, 'call:existsSync': 65, 'call:Lexer': 66, 'call:render': 67, 'call:encodeReg': 68, 'call:wrap': 69, 'call:_callee4': 70, 'precise:string.trim': 71, 'call:compileComparisonOp': 72, 'call:createFilter': 73, 'imprecise:assign': 74, 'call:start': 75, 'call:toString': 76, 'call:PTR': 77, 'imprecise:toString': 78, 'call:createPageConstructor': 79, 'call:createPageCache': 80, 'call:resolveDefs': 81, 'precise:string.substring': 82, 'call:execAsync': 83, 'imprecise:shift': 84, 'call:__constructor': 85, 'imprecise:String': 86, 'call:template': 87, 'imprecise:match': 88, 'call:match': 89, 'call:forEach': 90, 'call:charCodeAt': 91, 'call:isExpression': 92, 'call:compileJsExpression': 93, 'call:getExtensionChecker': 94, 'call:createBaseLoader': 95, 'call:createCompilerLoader': 96, 'call:__exportStar': 97, 'call:add': 98, 'call:unshift': 99}

# Mapping for sink types
sink_t_mapping = {
    'spawn': 0,
    'exec': 1,
    'Function': 2,
    'eval': 3,
}

# Mapping for vulnerability types
vuln_t_mapping = {
    'ACI': 0,
    'ACE': 1,
}

# For 'others' and 'unknown' categories
others_class = 100
unknown_class = 101

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--graph-out", type=str, required=True, help="Path to save the DGL graphs")
    parser.add_argument("--matrix-out", type=str, required=True, help="Path to save the feature matrix CSV file")
    parser.add_argument("--basic-meta-csv", type=str, required=True, help="Path to save the basic metadata CSV file")
        
    args = parser.parse_args()
    
    print("======= Building metadata =======")
    with open(args.dataset, 'r') as f:
        entries = json.load(f)["flows"]
        
        
    # 1. Writing the first CSV file (basic metadata with graph filename)
    with open(args.basic_meta_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'version', 'confirmed', 'vuln_type', 'graph_idx'])
        
        for idx, entry in enumerate(entries):            
            writer.writerow([entry['id'], entry['version'], entry['exploitable'] or entry["confirmed"], entry['vuln_type'], idx])
    
    
    print("======= Building graphs =======")
    pyg_graphs = []
    matrix_data = []
    total_vuln = 0
    total_graphs = 0
    node_id_counter = 0

    for idx, entry in enumerate(entries):
        node_ids = list(entry['provenance_tree'].keys())
        node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)

        src, dst = [], []
        ops_features, sink_t_features, tainted_features = [], [], []

        vuln_type = entry.get('vuln_type', '')
        if vuln_type in vuln_t_mapping:
            vult_t_value = vuln_t_mapping[vuln_type]
        elif vuln_type == '':
            vult_t_value = unknown_class
        else:
            vult_t_value = others_class
        vult_t_features = [vult_t_value] * num_nodes

        for node_idx, node_info in entry['provenance_tree'].items():
            cur_id = node_id_map[node_idx]
            op = node_info.get("operation", "")
            sink = node_info.get("sink_type", "")
            taint = int(node_info.get("tainted", False))

            ops_features.append(ops_mapping.get(op, unknown_class if op == "" else others_class))
            sink_t_features.append(sink_t_mapping.get(sink, unknown_class if sink == "" else others_class))
            tainted_features.append(taint)

            for parent in node_info.get("flows_from", []):
                if parent in node_id_map:
                    src.append(node_id_map[parent])
                    dst.append(cur_id)

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        graph = Data(edge_index=edge_index)
        graph.num_nodes = num_nodes

        graph.ops = torch.tensor(ops_features, dtype=torch.long)
        graph.sink_t = torch.tensor(sink_t_features, dtype=torch.long)
        graph.tainted = torch.tensor(tainted_features, dtype=torch.long)
        graph.vult_t = torch.tensor(vult_t_features, dtype=torch.long)
        graph.G_IDX = torch.full((num_nodes,), idx, dtype=torch.long)
        graph.N_IDX = torch.arange(node_id_counter, node_id_counter + num_nodes, dtype=torch.long)
        graph.VULN = torch.full((num_nodes,), int(entry["exploitable"]), dtype=torch.long)
        graph.y = graph.VULN
        graph.num_nodes = num_nodes
        
        exploit_val = 1.0 if entry["exploitable"] else 0.0
        
        if entry["exploitable"]:
            total_vuln += 1

        pyg_graphs.append(graph)
        total_graphs += 1
        node_id_counter += num_nodes

        for i in range(num_nodes):
            matrix_data.append({
                "graph_id": idx,
                "node_id": i,
                "operation": ops_features[i],
                "sink_type": sink_t_features[i],
                "tainted": tainted_features[i],
                "vult_t": vult_t_features[i],
                "vuln_or_not": int(exploit_val)
            })

    torch.save(pyg_graphs, args.graph_out)
    pd.DataFrame(matrix_data).to_csv(args.matrix_out, index=False)

    print(f"Total graphs: {total_graphs}, Total vulnerabilities: {total_vuln}")
    print(f"Feature matrix saved to {args.matrix_out}")
