"""Run OOD evaluation: essayset 1 = 15 epochs (already done), essaysets 2-8 = 10 epochs.

Usage:
    cd d:/LLMs/workspace/aitext
    python -u run_15epochs.py

Results saved to models/15epochs/ (existing results are preserved and appended to).
"""
import sys, os, time, json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
from torch.optim import AdamW
from src.model import LinearCRF
from src.dataset import get_ood_dataloaders
from src.train import train_epoch
from src.evaluate import evaluate_model

OUTPUT_DIR = os.path.join(ROOT, 'models', '15epochs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Output: {OUTPUT_DIR}', flush=True)

# Load existing results (essayset 1, already done)
results_path = os.path.join(OUTPUT_DIR, 'results_all_epochs.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        prev = json.load(f)
    results = prev.get('results', {})
    all_epochs = prev.get('all_epochs', {})
    done = {int(k) for k in results.keys()}
    print(f'Loaded existing: essaysets {sorted(done)} ({len(all_epochs.get("1",[]))} epochs each)')
else:
    results = {}
    all_epochs = {}
    done = set()

total_start = time.time()

for target_id in range(1, 9):
    num_epochs = 15 if target_id == 1 else 10

    if target_id in done:
        print(f'\nessayset {target_id}: already complete, skipping')
        continue

    print(f'\n{"="*50}')
    print(f'OOD essayset {target_id} / 8  ({num_epochs} epochs)')
    print(f'{"="*50}', flush=True)

    start = time.time()

    train_loader, test_loader = get_ood_dataloaders(
        feature_dir=os.path.join(ROOT, 'features'),
        target_prompt_id=target_id, batch_size=32
    )

    model = LinearCRF(input_dim=768, num_tags=2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epoch_log = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_model(model, test_loader, device, k_boundary=5)
        epoch_log.append({
            'epoch': epoch,
            'loss': float(train_loss),
            'token_f1': float(metrics['token_f1']),
            'token_precision': float(metrics['token_precision']),
            'token_recall': float(metrics['token_recall']),
            'boundary_f1_at_k': float(metrics['boundary_f1_at_k']),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Ep {epoch:2d} | Loss {train_loss:9.4f} | F1 {metrics["token_f1"]:.4f} | '
                  f'Prec {metrics["token_precision"]:.4f} | Rec {metrics["token_recall"]:.4f} | '
                  f'BndF1@5 {metrics["boundary_f1_at_k"]:.4f}', flush=True)

    elapsed = time.time() - start

    # Save model checkpoint
    model_path = os.path.join(OUTPUT_DIR, f'linear_crf_essayset_{target_id}.pt')
    torch.save(model.state_dict(), model_path)

    final = epoch_log[-1]
    results[str(target_id)] = {
        'train_n': len(train_loader.dataset),
        'test_n': len(test_loader.dataset),
        'epochs': num_epochs,
        'loss': final['loss'],
        'token_f1': final['token_f1'],
        'token_precision': final['token_precision'],
        'token_recall': final['token_recall'],
        'boundary_f1_at_k': final['boundary_f1_at_k'],
        'time_min': elapsed / 60,
    }
    all_epochs[str(target_id)] = epoch_log

    print(f'  Saved {model_path} ({elapsed/60:.1f} min)', flush=True)

    # Save after each essayset (preserves existing data, appends new)
    save_data = {
        'config': {'batch_size': 32, 'lr': 1e-4, 'weight_decay': 1e-2},
        'results': results,
        'all_epochs': all_epochs,
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)

# -------- Final summary --------
total_time = (time.time() - total_start) / 60

summary_lines = [
    f'{"="*65}',
    f'FINAL RESULTS ({total_time:.0f} min total)',
    f'{"="*65}',
    f'{"Test":<6} {"Epochs":<7} {"TrainN":<8} {"TestN":<7} {"Loss":<9} {"TokenF1":<9} {"Prec":<9} {"Recall":<9} {"BndF1@5":<9}',
    '-' * 65,
]

for tid in range(1, 9):
    r = results[str(tid)]
    summary_lines.append(
        f'{tid:<6} {r["epochs"]:<7} {r["train_n"]:<8} {r["test_n"]:<7} '
        f'{r["loss"]:<9.4f} {r["token_f1"]:<9.4f} {r["token_precision"]:<9.4f} '
        f'{r["token_recall"]:<9.4f} {r["boundary_f1_at_k"]:<9.4f}'
    )

for line in summary_lines:
    print(line, flush=True)

with open(os.path.join(OUTPUT_DIR, 'results_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nAll results saved to: {OUTPUT_DIR}', flush=True)