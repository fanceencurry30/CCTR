#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import lmdb
from tqdm import tqdm

def create_line_lmdb(input_lmdb_path, output_lmdb_path):
    """Convert whole image LMDB to line text LMDB (process all annotation types)"""
    if not os.path.exists(input_lmdb_path):
        raise FileNotFoundError(f"The input LMDB path does not exist: {input_lmdb_path}")

    os.makedirs(output_lmdb_path, exist_ok=True)
    
    # Iput LMDB
    env_in = lmdb.open(input_lmdb_path, readonly=True, max_readers=100, lock=False)
    
    # calculate LMDB size
    with env_in.begin() as txn:
        num_images = int(txn.get(b'num-samples').decode())
    map_size = num_images * 30 * 3 * 1024 * 10 * 2  

    env_out = lmdb.open(output_lmdb_path, map_size=map_size)

    samples_processed = 0

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        
        pbar = tqdm(total=num_images, desc="Processing images")
        
        for key, value in txn_in.cursor():
            if not key.startswith(b'image-'):
                continue

            # data
            image_id = int(key.decode().split('-')[1])
            label_key = f'label-{image_id:09d}'.encode()
            label_data = txn_in.get(label_key)
            
            if not label_data:
                pbar.update(1)
                continue

            # Decode images and annotations
            img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            label_info = json.loads(label_data.decode('utf-8'))
            
            # Get all text lines for all annotation types
            text_lines = []
            for annotation_type, lines in label_info['annotations'].items():
                if isinstance(lines, list):
                    # Add annotation type information for each line for subsequent identification
                    for line in lines:
                        line['annotation_type'] = annotation_type
                    text_lines.extend(lines)
            
            # Skip this image if there are no text lines
            if not text_lines:
                pbar.update(1)
                continue
            
            # Sort by vertical position of text line (from top to bottom)
            text_lines.sort(key=lambda x: np.mean([x['point'][i] for i in range(1, 8, 2)]))  # Take the average of all y coordinates
            
            # Process each line of text
            for line_idx, line in enumerate(text_lines, 1):
                try:
                    # Analytical quadrilateral coordinates (4 points, x, y alternating for each point)
                    pts = np.array(line['point'], dtype=np.float32).reshape(4, 2)
                    
                    # Calculate clipping rectangle
                    x, y, w, h = cv2.boundingRect(pts)
                    line_img = img[y:y+h, x:x+w]
                    
                    # encode JPEG
                    _, img_enc = cv2.imencode('.jpg', line_img)
                    if img_enc is None:
                        continue
                    
                    # Generate a label with metadata (including the original annotation type)
                    clean_text = line['text'].strip()
                    annotation_type = line.get('annotation_type', 'unknown')
                    new_label = f"{clean_text} <image_id={image_id}_line_id={line_idx}>"
                    
                    # save to new LMDB
                    new_id = samples_processed + 1
                    txn_out.put(f'image-{new_id:09d}'.encode(), img_enc.tobytes())
                    txn_out.put(f'label-{new_id:09d}'.encode(), new_label.encode('utf-8'))
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"Error processing picture {image\u id} line {line\u idx}: {str(e)}")
                    continue
            
            pbar.update(1)
        pbar.close()

    # Total number of samples
    with env_out.begin(write=True) as txn_final:
        txn_final.put(b'num-samples', str(samples_processed).encode())

    env_in.close()
    env_out.close()

    print(f"\n Conversion complete: {output_lmdb_path}")
    print(f"A total of {num\u images} original images are processed")
    print(f"Generate {samples_processed} line text samples")

if __name__ == "__main__":
    input_lmdbs = [

        ['./train_data/hccdoc_test_lmdb']
    ]
    output_lmdbs = [

        ['./image_hw_lmdb']
    ]
    print(len(input_lmdbs))
    for i in range(len(input_lmdbs)):
        input_lmdb = input_lmdbs[i][0]
        output_lmdb = output_lmdbs[i][0]
        create_line_lmdb(input_lmdb, output_lmdb)