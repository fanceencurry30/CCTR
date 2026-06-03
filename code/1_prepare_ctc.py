import os
import json
import torch
from collections import defaultdict
import numpy as np
from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger
        
def parse_label(label_str):
    """Parse label string with metadata, e.g., 'hello <image_id=1_line_id=2>'"""
    if '<image_id=' in label_str and '_line_id=' in label_str:
        parts = label_str.rsplit('<', 1)
        text = parts[0].strip()
        meta_str = '<' + parts[1]
        image_id = int(meta_str.split('image_id=')[1].split('_')[0])
        line_id = int(meta_str.split('line_id=')[1].split('>')[0])
        return text, image_id, line_id
    return label_str, -1, -1

def ctc_filter(topk_probs, topk_indices):
    """Filter topk results according to CTC rules: remove blank and consecutive duplicate characters"""
    filtered_probs = []
    filtered_indices = []
    prev_index = None 

    # Iterate through each time step
    for t in range(len(topk_indices)):
        current_index = topk_indices[t][0]  # Take top1 index
        if current_index != 0:  # Skip blank
            if current_index != prev_index:  # Skip consecutive duplicates
                filtered_probs.append(topk_probs[t].tolist())
                filtered_indices.append(topk_indices[t].tolist())
        prev_index = current_index
    return filtered_probs, filtered_indices

def generate_lister_ctc_data(config_path, model_path, output_dir):
    # Load configuration
    cfg = Config(config_path).cfg
    logger = get_logger('generate_lister_ctc_data', os.path.join(output_dir, 'generate_lister_ctc_data.log'))

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process_class.get_character_num()

    # Set output channels
    if 'lister_decoder' in cfg['Architecture']['Decoder']:
        cfg['Architecture']['Decoder']['lister_decoder']['out_channels'] = char_num
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    model = build_rec_model(cfg['Architecture'])
    load_ckpt(model, cfg, None, None)
    model.to(device)
    model.eval()

    # Build dataloader
    cfg['Global']['distributed'] = False
    dataloader = build_dataloader(cfg, 'Eval', logger, task='rec')

    os.makedirs(output_dir, exist_ok=True)

    # Main aggregation structure
    image_results = defaultdict(lambda: {
        'lines': [],  # Store complete structure for each line
    })

    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        batch_tensor = [t.to(device) for t in batch]
        batch_numpy = [t.cpu().numpy() for t in batch]

        with torch.no_grad():
            preds = model(batch_tensor[0], data=batch_tensor[1:])

            # Get logits
            if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], dict) and 'logits' in preds[1]:
                ctc_logits = preds[1]['logits'][-1]
                ctc_probs = torch.softmax(ctc_logits, dim=-1)
            elif isinstance(preds, list) and preds and isinstance(preds[-1], dict) and 'logits' in preds[-1]:
                ctc_logits = preds[-1]['logits']
                ctc_probs = torch.softmax(ctc_logits, dim=-1)
            else :
                ctc_probs = preds

            topk_probs, topk_indices = torch.topk(ctc_probs, k=100, dim=-1, largest=True, sorted=True)

            # Decode text
            decoded_texts, labels = post_process_class(preds, batch_numpy)
            decoded_texts = [text[0] if isinstance(text, tuple) else text for text in decoded_texts]

            for i in range(len(labels)):
                label_text = labels[i][0] if isinstance(labels[i], (list, tuple)) else labels[i]
                text, image_id, line_id = parse_label(label_text)
                if image_id == -1:
                    continue
                
                # Apply CTC filtering
                filtered_probs, filtered_indices = ctc_filter(
                    topk_probs[i].cpu().numpy(),
                    topk_indices[i].cpu().numpy()
                )
                
                line_data = {
                    'line_id': line_id,
                    'decoded_text': decoded_texts[i],
                    'label': text,
                    'label_len': len(text),
                    'topk_probs': filtered_probs,  # Use filtered probabilities
                    'topk_indices': filtered_indices,  # Use filtered indices
                    'batch_idx': batch_idx,
                    'original_timesteps': len(topk_indices[i]),
                    'filtered_timesteps': len(filtered_indices)  
                }
                image_results[image_id]['lines'].append(line_data)

    # Organize output results
    final_results = []
    for img_id, data in image_results.items():
        # Sort by line ID and batch order
        lines_sorted = sorted(data['lines'], key=lambda x: (x['line_id'], x['batch_idx']))
        
        # Calculate statistics
        total_timesteps = sum(line['filtered_timesteps'] for line in lines_sorted) 
        avg_timesteps_per_line = total_timesteps / len(lines_sorted) if lines_sorted else 0
        
        # Concatenate complete text
        merged_decoded_text = "".join([line['decoded_text'] for line in lines_sorted])
        merged_decoded_text_len = len(merged_decoded_text)  # New: predicted text length
        merged_label = "".join([line['label'] for line in lines_sorted])
        
        # Ensure all topk_probs are 2D arrays with second dimension 100
        prob_arrays = []
        for line in lines_sorted:
            prob_array = np.array(line['topk_probs'])
            if prob_array.size == 0:  # Skip if empty after filtering
                continue
            if prob_array.ndim == 1:
                prob_array = prob_array.reshape(1, -1)
            prob_arrays.append(prob_array)
        
        # Create empty 2D array if all lines are empty
        if not prob_arrays:
            merged_probs = np.zeros((0, 100))
        else:
            merged_probs = np.concatenate(prob_arrays, axis=0)
        
        # Apply same processing to topk_indices
        index_arrays = []
        for line in lines_sorted:
            index_array = np.array(line['topk_indices'])
            if index_array.size == 0:  # Skip if empty after filtering
                continue
            if index_array.ndim == 1:
                index_array = index_array.reshape(1, -1)
            index_arrays.append(index_array)
        
        if not index_arrays:
            merged_indices = np.zeros((0, 100), dtype=int)
        else:
            merged_indices = np.concatenate(index_arrays, axis=0)
        
        
        result = {
            'image_id': img_id,
            'decoded_text': merged_decoded_text,
            'decoded_text_len': merged_decoded_text_len,
            'label': merged_label,
            'num_lines': len(lines_sorted),
            'total_timesteps': total_timesteps,
            'avg_timesteps_per_line': round(avg_timesteps_per_line, 2),
            'label_len': len(merged_label),
            'topk_probs': merged_probs.tolist(),
            'topk_indices': merged_indices.tolist(),
            'line_ids': [line['line_id'] for line in lines_sorted],
            'timesteps_to_text_ratio': round(total_timesteps / merged_decoded_text_len, 2) if merged_decoded_text_len > 0 else 0
        }
        final_results.append(result)

        # Save per-image results
        output_path = os.path.join(output_dir, f'image_{img_id}_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Results for image {img_id} saved ({result['num_lines']} lines of text)")

    # Summary information
    final_results.sort(key=lambda x: x['image_id'])
    summary = {
        'total_images': len(final_results),
        'total_lines': sum(r['num_lines'] for r in final_results),
        'output_dir': os.path.abspath(output_dir)
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Processing complete! Total {summary['total_images']} images and {summary['total_lines']} lines of text processed")
    return final_results


if __name__ == "__main__":
    config_path = './method_yml/crnn.yml'
    model_path = './method_pth/best_crnn.pth'
    output_dir = './ctc_probs/ctc_probs_crnn_train_filter'
    generate_lister_ctc_data(config_path, model_path, output_dir)
    print(f"✅ Top-100 CTC probabilities and decoded text generated and saved to {output_dir}")