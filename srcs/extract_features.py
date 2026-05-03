"""
Stage 1: Offline Feature Extraction
Extract DeBERTa last_hidden_state features from hybrid essays.
Saves features grouped by essayset into HDF5 files (one per essayset).
"""

import os
import ast
import torch
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer


def load_deberta_model(model_dir: str, device: str = 'cuda') -> Tuple:
    """Load DeBERTa model and tokenizer from local directory, move to device."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_data(excel_path: str) -> pd.DataFrame:
    """Load data from Excel file."""
    return pd.read_excel(excel_path)


def align_labels_to_subwords(
    sent_and_label: List,
    hybrid_text: str,
    tokenizer,
    max_len: int = 512
) -> np.ndarray:
    """Broadcast sentence-level labels to subword tokens."""
    labels = np.full(max_len, -100, dtype=np.int16)

    char_labels = []
    for sentence, label_str in sent_and_label:
        label = 0 if label_str == "human" else 1
        char_labels.extend([label] * len(sentence))

    encoding = tokenizer(
        hybrid_text,
        return_offsets_mapping=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors=None
    )

    offset_mapping = encoding['offset_mapping']

    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        if token_idx >= max_len:
            break
        if start_char == 0 and end_char == 0:
            labels[token_idx] = -100
        elif start_char < len(char_labels):
            labels[token_idx] = char_labels[start_char]

    return labels


def extract_batch(
    batch_texts: List[str],
    batch_sent_and_labels: List,
    model,
    tokenizer,
    device: str,
    max_len: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for a batch of essays on GPU.

    Returns:
        (features [B, 512, 768] float16, labels [B, 512] int16, masks [B, 512] uint8)
    """
    # Tokenize batch
    encodings = tokenizer(
        batch_texts,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Align labels for each essay in the batch
    batch_labels = []
    for sent_and_label, text in zip(batch_sent_and_labels, batch_texts):
        labels = align_labels_to_subwords(sent_and_label, text, tokenizer, max_len)
        batch_labels.append(labels)

    # Extract features on GPU
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state.cpu().numpy().astype(np.float16)

    return (
        features,
        np.stack(batch_labels),
        attention_mask.cpu().numpy().astype(np.uint8)
    )


def extract_all_features(
    data_path: str,
    model_dir: str,
    output_dir: str,
    max_len: int = 512,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """Extract features for all essays, save grouped by essayset as HDF5."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Loading DeBERTa model from {model_dir} to {device}...")
    model, tokenizer = load_deberta_model(model_dir, device)
    print("Model loaded successfully.")

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} essays.")

    os.makedirs(output_dir, exist_ok=True)

    # Pre-parse sent_and_label
    sent_and_labels = []
    for _, row in df.iterrows():
        raw = row['sent_and_label']
        sent_and_labels.append(
            ast.literal_eval(raw) if isinstance(raw, str) else raw
        )

    # Group row indices by essayset
    essayset_groups = defaultdict(list)
    for idx, row in df.iterrows():
        essayset_groups[int(row['essayset'])].append(idx)

    print(f"Found {len(essayset_groups)} essaysets")

    # Process each essayset
    for essayset, indices in sorted(essayset_groups.items()):
        total = len(indices)
        print(f"\nProcessing essayset {essayset} ({total} essays, batch_size={batch_size})...")

        feature_chunks = []
        label_chunks = []
        mask_chunks = []
        essay_ids_list = []

        # Process in batches
        for i in range(0, total, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_texts = [str(df.iloc[idx]['hybrid_text']) for idx in batch_indices]
            batch_sals = [sent_and_labels[idx] for idx in batch_indices]
            batch_ids = [int(df.iloc[idx]['essay_id']) for idx in batch_indices]

            feats, labs, masks = extract_batch(
                batch_texts, batch_sals, model, tokenizer, device, max_len
            )

            feature_chunks.append(feats)
            label_chunks.append(labs)
            mask_chunks.append(masks)
            essay_ids_list.extend(batch_ids)

            if (i // batch_size + 1) % 20 == 0:
                done = i + len(batch_indices)
                print(f"  {done}/{total} samples processed")

        # Concatenate batches for this essayset
        all_features = np.concatenate(feature_chunks, axis=0)
        all_labels = np.concatenate(label_chunks, axis=0)
        all_masks = np.concatenate(mask_chunks, axis=0)
        all_essay_ids = np.array(essay_ids_list, dtype=np.int32)

        # Write to HDF5
        output_path = os.path.join(output_dir, f"essayset_{essayset}.h5")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=all_features, dtype='float16')
            f.create_dataset('labels', data=all_labels, dtype='int16')
            f.create_dataset('attention_mask', data=all_masks, dtype='uint8')
            f.create_dataset('essay_ids', data=all_essay_ids, dtype='int32')
            f.attrs['essayset'] = essayset
            f.attrs['num_samples'] = total

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved {output_path} ({size_mb:.1f} MB, {total} samples)")

    print(f"\nFeature extraction complete. Saved {len(essayset_groups)} HDF5 files to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract DeBERTa features from essays')
    parser.add_argument('--data_path', type=str, default='data/data.xlsx',
                        help='Path to data Excel file')
    parser.add_argument('--model_dir', type=str, default='deberta',
                        help='Path to DeBERTa model directory')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='Output directory for feature files')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for GPU inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    extract_all_features(
        data_path=args.data_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()