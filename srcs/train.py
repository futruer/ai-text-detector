"""
Training loop for Linear-CRF model with OOD evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from .model import LinearCRF
from .dataset import get_ood_dataloaders
from .evaluate import evaluate_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float = 1.0
) -> float:
    """Train for one epoch. Returns average training loss."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        loss = model(features, labels, mask)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Linear-CRF model')
    parser.add_argument('--feature_dir', type=str, default='features',
                        help='Directory containing essayset_*.h5 feature files')
    parser.add_argument('--target_prompt_id', type=int, default=1,
                        help='Essayset ID to hold out for testing (OOD)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--k_boundary', type=int, default=5,
                        help='K for Boundary F1@K metric')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create OOD dataloaders
    print(f"\nCreating OOD dataloaders (target_prompt_id={args.target_prompt_id})...")
    train_loader, test_loader = get_ood_dataloaders(
        feature_dir=args.feature_dir,
        target_prompt_id=args.target_prompt_id,
        batch_size=args.batch_size
    )

    # Initialize model
    print("\nInitializing Linear-CRF model...")
    model = LinearCRF(input_dim=768, num_tags=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer: AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            gradient_clip_norm=args.gradient_clip_norm
        )
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluate
        metrics = evaluate_model(model, test_loader, device, k_boundary=args.k_boundary)
        print(f"  Token F1: {metrics['token_f1']:.4f}")
        print(f"  Token Precision: {metrics['token_precision']:.4f}")
        print(f"  Token Recall: {metrics['token_recall']:.4f}")
        print(f"  Boundary F1@{args.k_boundary}: {metrics['boundary_f1_at_k']:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'linear_crf_essayset_{args.target_prompt_id}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()