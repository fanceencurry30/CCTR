import csv
import os
import sys
import numpy as np
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from openrec.metrics import build_metric
from tqdm import tqdm

def eval_image_acc(folder_path):
    config = {'name': 'RecMetric'}
    metric = build_metric(config)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} files")
    print("evaluating:", folder_path)
    
    # Collect all predictions and labels
    all_preds = []
    all_labels = []
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(folder_path, json_file)

        try:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error opening file {json_file}: {str(e)}")
            continue            
        
        pred = [(data['decoded_texts'][0], 1.0)]
        label = [(data['labels'][0][0], 1.0)]
        pred_label = [pred, label]
        evaling = metric.eval_all_metric(pred_label)

    
    metrics = metric.get_all_metric()
    print("acc:", metrics['acc'])
    print("ar:", metrics['ar'])
    print("cr:", metrics['cr'])
    
if __name__ == "__main__":
    # Dataset to evaluate
    input_dirs_double = "./ctc_probs/lister_test/"

    input_dirs = os.listdir(input_dirs_double)
    print(input_dirs)
    for i in range(len(input_dirs)):
        eval_image_acc(os.path.join(input_dirs_double, input_dirs[i]))