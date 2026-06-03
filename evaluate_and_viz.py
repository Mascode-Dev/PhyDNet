"""Comprehensive evaluation and visualization for PhyDNet with/without velocity.

Scans Moving MNIST test set to find worst-case sequences where velocity advection
matters most, generates diff maps, per-frame MSE curves, and prediction grids.

Outputs everything to figures/."""

import torch
import numpy as np
import os
import time
from collections import defaultdict
from models.models import ConvLSTM, PhyCell, EncoderRNN
from data.moving_mnist import MovingMNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

TEST_BATCH = 16
MAX_BATCHES = 400
PRED_FRAMES = 10
NUM_WORST = 3


def build_model():
    phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                      n_layers=1, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                        n_layers=3, kernel_size=(3, 3), device=device)
    return EncoderRNN(phycell, convcell, device)


def predict_sequence(encoder, input_tensor, input_velocity, output_velocity, use_velocity=True):
    with torch.no_grad():
        input_length = input_tensor.size(1)
        target_length = output_velocity.size(1)

        for ei in range(input_length - 1):
            vel = input_velocity[:, ei, :] if use_velocity else None
            encoder(input_tensor[:, ei, :, :, :], (ei == 0), velocity=vel)

        decoder_input = input_tensor[:, -1, :, :, :]
        predictions = []
        for di in range(target_length):
            vel = output_velocity[:, di, :] if use_velocity else None
            _, _, output_image, _, _ = encoder(decoder_input, False, False, velocity=vel)
            decoder_input = output_image
            predictions.append(output_image.cpu().numpy())

        return np.stack(predictions).swapaxes(0, 1)


def compute_per_sample_mse(preds, targets):
    """preds: (B, T, 1, 64, 64), targets: (B, T, 1, 64, 64) -> mse per (B, T)"""
    return np.mean((preds - targets) ** 2, axis=(2, 3, 4))


def save_prediction_grid(seq_idx, gt, pred_w, pred_n, diff_w, diff_n, filename, frames_to_show=10):
    """4 rows: GT, With vel, No vel, Diff map. Saves full sequence."""
    nrows, ncols = 4, frames_to_show
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))

    # Normalize diff maps for fair comparison
    all_diffs = np.concatenate([diff_w[:frames_to_show], diff_n[:frames_to_show]])
    diff_max = np.percentile(all_diffs, 98)

    for c in range(frames_to_show):
        axes[0, c].imshow(gt[c].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, c].imshow(pred_w[c].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[2, c].imshow(pred_n[c].squeeze(), cmap='gray', vmin=0, vmax=1)
        im = axes[3, c].imshow(diff_n[c].squeeze() - diff_w[c].squeeze(), cmap='RdBu_r',
                               vmin=-diff_max, vmax=diff_max, interpolation='bilinear')

    row_labels = ['Ground Truth', 'With Velocity', 'Without Velocity', 'Error Diff\n(NoVel - Vel)']
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8)
    for c in range(frames_to_show):
        axes[0, c].set_title(f't+{c + 1}', fontsize=8)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    cbar = fig.colorbar(im, ax=axes[3, 0], orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label('MSE diff', fontsize=7)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_mse_curve(frame_mse_w, frame_mse_n, filename):
    """Per-frame MSE comparison curve across 10 prediction steps."""
    fig, ax = plt.subplots(figsize=(6, 4))
    frames = np.arange(1, PRED_FRAMES + 1)
    ax.plot(frames, frame_mse_w, 'o-', color='#2ca02c', linewidth=2, markersize=6, label='With Velocity')
    ax.plot(frames, frame_mse_n, 's-', color='#d62728', linewidth=2, markersize=6, label='Without Velocity')
    ax.fill_between(frames, frame_mse_n, frame_mse_w, alpha=0.15, color='#2ca02c')
    ax.set_xlabel('Prediction Timestep', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Per-Frame Prediction Error (Moving MNIST Test Set)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(frames)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_compact_poster_grid(seq_idx, gt, pred_w, pred_n, diff_n, diff_w, filename):
    """Compact 3-row poster view: GT + With Vel + Without Vel, with diff overlay."""
    ncols = 5
    nrows = 3
    times = [0, 2, 4, 7, 9]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))

    for ci, ti in enumerate(times):
        axes[0, ci].imshow(gt[ti].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, ci].imshow(pred_w[ti].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[2, ci].imshow(pred_n[ti].squeeze(), cmap='gray', vmin=0, vmax=1)

    row_labels = ['Ground Truth', 'With Velocity', 'Without Velocity']
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=9)
    for ci, ti in enumerate(times):
        axes[0, ci].set_title(f't+{ti + 1}', fontsize=9)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_metrics_table(mse_w_avg, ssim_w, mse_n_avg, ssim_n, filename):
    """Generate a clean metrics table as a figure."""
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('tight')
    ax.axis('off')

    improvement_mse = (mse_n_avg - mse_w_avg) / mse_n_avg * 100
    improvement_ssim = ssim_w - ssim_n

    table_data = [
        ['', 'MSE', 'SSIM'],
        ['With Velocity', f'{mse_w_avg:.4f}', f'{ssim_w:.4f}'],
        ['Without Velocity', f'{mse_n_avg:.4f}', f'{ssim_n:.4f}'],
        ['Improvement', f'-{improvement_mse:.1f}%', f'+{improvement_ssim:.3f}'],
    ]
    colors = [['#f0f0f0', '#f0f0f0', '#f0f0f0'],
              ['white', 'white', '#e8f5e9'],
              ['white', 'white', '#ffebee'],
              ['#f0f0f0', '#e8f5e9', '#e8f5e9']]

    table = ax.table(cellText=table_data, cellColours=colors, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


print("Loading models...")
encoder_w = build_model()
encoder_w.load_state_dict(torch.load('save/encoder_with_velocity_e99.pth', map_location=device))
encoder_w.eval()

encoder_n = build_model()
encoder_n.load_state_dict(torch.load('save/encoder_no_velocity_e99.pth', map_location=device))
encoder_n.eval()

mm_test = MovingMNIST(root='data/', is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mm_test, batch_size=TEST_BATCH, shuffle=False, num_workers=0)

print("Scanning test set for worst-case sequences...")
sample_errors = []
running_mse_w = np.zeros(PRED_FRAMES)
running_mse_n = np.zeros(PRED_FRAMES)
running_ssim_w_total = 0
running_ssim_n_total = 0
n_samples = 0

from skimage.metrics import structural_similarity as ssim

t0 = time.time()
for batch_idx, out in enumerate(test_loader):
    if batch_idx >= MAX_BATCHES:
        break

    inp = out[1].to(device)
    tgt = out[2].to(device)
    inv = out[3].to(device)
    outv = out[4].to(device)

    target_np = tgt.cpu().numpy()

    preds_w = predict_sequence(encoder_w, inp, inv, outv, use_velocity=True)
    preds_n = predict_sequence(encoder_n, inp, inv, outv, use_velocity=False)

    mse_w = compute_per_sample_mse(preds_w, target_np)
    mse_n = compute_per_sample_mse(preds_n, target_np)

    gap = mse_n - mse_w

    for s in range(TEST_BATCH):
        avg_gap = gap[s].mean()
        tot_mse_w = mse_w[s].mean()
        tot_mse_n = mse_n[s].mean()
        sample_errors.append({
            'global_idx': batch_idx * TEST_BATCH + s,
            'batch_idx': batch_idx,
            'local_idx': s,
            'gap': avg_gap,
            'mse_w': tot_mse_w,
            'mse_n': tot_mse_n,
            'per_frame_w': mse_w[s],
            'per_frame_n': mse_n[s],
            'input': inp[s:s + 1].clone(),
            'target': tgt[s:s + 1].clone(),
            'inv': inv[s:s + 1].clone(),
            'outv': outv[s:s + 1].clone(),
        })

    running_mse_w += mse_w.sum(axis=0)
    running_mse_n += mse_n.sum(axis=0)
    n_samples += TEST_BATCH

    for a in range(TEST_BATCH):
        for f in range(PRED_FRAMES):
            running_ssim_w_total += ssim(target_np[a, f, 0], preds_w[a, f, 0], data_range=1.0)
            running_ssim_n_total += ssim(target_np[a, f, 0], preds_n[a, f, 0], data_range=1.0)

    if (batch_idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Batch {batch_idx + 1}/{MAX_BATCHES} ({elapsed:.0f}s)")

frame_mse_w = running_mse_w / n_samples
frame_mse_n = running_mse_n / n_samples
avg_mse_w = frame_mse_w.mean()
avg_mse_n = frame_mse_n.mean()
avg_ssim_w = running_ssim_w_total / (n_samples * PRED_FRAMES)
avg_ssim_n = running_ssim_n_total / (n_samples * PRED_FRAMES)

print(f"\nResults over {n_samples} samples:")
print(f"  With velocity:    MSE={avg_mse_w:.4f}, SSIM={avg_ssim_w:.4f}")
print(f"  Without velocity: MSE={avg_mse_n:.4f}, SSIM={avg_ssim_n:.4f}")
print(f"  Improvement:      MSE={avg_mse_n - avg_mse_w:.4f} ({(avg_mse_n - avg_mse_w) / avg_mse_n * 100:.1f}%), SSIM=+{avg_ssim_w - avg_ssim_n:.4f}")

sample_errors.sort(key=lambda x: -x['gap'])
worst = sample_errors[:NUM_WORST]

print(f"\nGenerating figures...")

save_mse_curve(frame_mse_w, frame_mse_n, 'figures/per_frame_mse.png')
save_metrics_table(avg_mse_w, avg_ssim_w, avg_mse_n, avg_ssim_n, 'figures/metrics_table.png')

for rank, item in enumerate(worst):
    global_idx = item['global_idx']
    gap_avg = item['gap']

    inp_t = item['input'].to(device)
    tgt_t = item['target'].to(device)
    inv_t = item['inv'].to(device)
    outv_t = item['outv'].to(device)

    target_np = tgt_t.cpu().numpy()[0]
    preds_w = predict_sequence(encoder_w, inp_t, inv_t, outv_t, use_velocity=True)[0]
    preds_n = predict_sequence(encoder_n, inp_t, inv_t, outv_t, use_velocity=False)[0]

    diff_w = np.abs(preds_w - target_np)
    diff_n = np.abs(preds_n - target_np)

    print(f"\n  Worst #{rank + 1}: sequence {global_idx}, avg gap={gap_avg:.6f} "
          f"(MSE w={item['mse_w']:.4f}, n={item['mse_n']:.4f})")

    save_prediction_grid(
        global_idx, target_np, preds_w, preds_n, diff_w, diff_n,
        f'figures/worst_case_{rank + 1}_full.png', frames_to_show=10
    )

    save_compact_poster_grid(
        global_idx, target_np, preds_w, preds_n, diff_n, diff_w,
        f'figures/worst_case_{rank + 1}_compact.png'
    )

print(f"\nDone. All figures saved to figures/")
