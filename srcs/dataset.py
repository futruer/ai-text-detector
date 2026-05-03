"""
OOD Dataset splitting and DataLoader for AI text detection.
Reads features from HDF5 files (one per essayset).
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


class FeatureDataset(Dataset):
    """PyTorch Dataset for pre-extracted features stored in HDF5 files.

    Keeps HDF5 file handles open for efficient random access.
    """

    def __init__(self, feature_dir: str, essayset_ids: List[int]):
        self.feature_dir = feature_dir
        self.samples = []
        self._files = {}  # Cache: h5_path -> h5py.File handle

        for essayset_id in essayset_ids:
            h5_path = os.path.join(feature_dir, f"essayset_{essayset_id}.h5")
            if not os.path.exists(h5_path):
                print(f"Warning: {h5_path} not found, skipping essayset {essayset_id}")
                continue

            f = h5py.File(h5_path, 'r')
            self._files[h5_path] = f
            num_samples = f.attrs['num_samples']
            for i in range(num_samples):
                self.samples.append((h5_path, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        h5_path, row_idx = self.samples[idx]
        f = self._files[h5_path]

        features = torch.from_numpy(f['features'][row_idx].astype(np.float32))
        labels = torch.from_numpy(f['labels'][row_idx].astype(np.int64))
        attention_mask = torch.from_numpy(f['attention_mask'][row_idx].astype(np.int64))
        essay_id = int(f['essay_ids'][row_idx])
        essayset_id = int(f.attrs['essayset'])

        return {
            'features': features,
            'labels': labels,
            'attention_mask': attention_mask,
            'essayset_id': essayset_id,
            'essay_id': essay_id
        }

    def close(self):
        """Close all open HDF5 handles."""
        for f in self._files.values():
            f.close()
        self._files.clear()


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    return {
        'features': torch.stack([item['features'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'mask': torch.stack([item['attention_mask'] for item in batch]),
        'essayset_ids': [item['essayset_id'] for item in batch],
        'essay_ids': [item['essay_id'] for item in batch]
    }


def get_available_essaysets(feature_dir: str) -> List[int]:
    """Get list of essayset IDs available in the feature directory."""
    essaysets = []
    for h5_file in glob.glob(os.path.join(feature_dir, 'essayset_*.h5')):
        basename = os.path.basename(h5_file)
        essayset_id = int(basename.replace('essayset_', '').replace('.h5', ''))
        essaysets.append(essayset_id)
    return sorted(essaysets)


def get_ood_dataloaders(
    feature_dir: str,
    target_prompt_id: int,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders for OOD evaluation.

    Args:
        feature_dir: Directory containing essayset_*.h5 feature files
        target_prompt_id: Essayset ID to hold out for testing
        batch_size: Batch size for DataLoader

    Returns:
        (train_loader, test_loader)
    """
    all_essaysets = get_available_essaysets(feature_dir)
    print(f"All essaysets: {all_essaysets}")
    print(f"Target prompt ID (test): {target_prompt_id}")

    # Train: all essaysets except target
    train_essaysets = [es for es in all_essaysets if es != target_prompt_id]
    # Test: only target essayset
    test_essaysets = [target_prompt_id]

    print(f"Train essaysets: {train_essaysets}")
    print(f"Test essaysets: {test_essaysets}")

    # Create datasets
    train_dataset = FeatureDataset(feature_dir, train_essaysets)
    test_dataset = FeatureDataset(feature_dir, test_essaysets)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader