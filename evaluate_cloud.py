"""Comprehensive evaluation and visualization for PhyDNet on GOES-16 cloud data.

Loads the trained cloud checkpoint, scans test set for worst-case sequences,
generates diff maps, per-frame MSE curve, prediction grids, and metrics.

Run with --compare to also train a no-velocity model and show side-by-side."""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import random
from data.goes_cloud import GOESCloud
from models.models import ConvLSTM, PhyCell, EncoderRNN
from constrain_moments import K2M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLOUD_DATE = '2024-07-14'
CLOUD_START = 18
CLOUD_HOURS = 6
CLOUD_INTERVAL = 5
PRED_FRAMES = 10
DO_COMPARE = '--compare' in sys.argv

os.makedirs('figures', exist_ok=True)


def build_model():
    phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                      n_layers=1, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                        n_layers=3, kernel_size=(3, 3), device=device)
    return EncoderRNN(phycell, convcell, device)


def predict_cloud(encoder, input_tensor, input_velocity, output_velocity, use_velocity=True):
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


def train_cloud_no_velocity(encoder, train_loader, nepochs=50):
    """Quick train a no-velocity model on cloud data."""
    constraints = torch.zeros((49, 7, 7)).to(device)
    ind = 0
    for i in range(7):
        for j in range(7):
            constraints[ind, i, j] = 1
            ind += 1

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(nepochs):
        loss_epoch = 0
        tf_ratio = max(0, 1 - epoch * 0.003)
        for out in train_loader:
            inp = out[1].to(device)
            tgt = out[2].to(device)
            inv = out[3].to(device)
            outv = out[4].to(device)

            optimizer.zero_grad()
            loss = 0
            for ei in range(inp.size(1) - 1):
                _, _, out_img, _, _ = encoder(inp[:, ei], (ei == 0))
                loss += criterion(out_img, inp[:, ei + 1])

            dec_in = inp[:, -1]
            use_tf = random.random() < tf_ratio
            for di in range(tgt.size(1)):
                _, _, out_img, _, _ = encoder(dec_in)
                loss += criterion(out_img, tgt[:, di])
                dec_in = tgt[:, di] if use_tf else out_img

            k2m = K2M([7, 7]).to(device)
            for b in range(encoder.phycell.cell_list[0].input_dim):
                filters = encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
                m = k2m(filters.double()).float()
                loss += criterion(m, constraints)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() / tgt.size(1)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{nepochs}  loss {loss_epoch:.2f}")

    return encoder


def save_cloud_prediction_grid(seq_idx, gt, pred, diff, filename, frames_to_show=10):
    nrows, ncols = 3, frames_to_show
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))

    for c in range(frames_to_show):
        axes[0, c].imshow(gt[c].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, c].imshow(pred[c].squeeze(), cmap='gray', vmin=0, vmax=1)
        im = axes[2, c].imshow(diff[c].squeeze(), cmap='hot', vmin=0, vmax=0.3)

    for r in range(nrows):
        axes[r, 0].set_ylabel(['Ground Truth', 'Predicted', '|Error|'][r], fontsize=8)
    for c in range(frames_to_show):
        axes[0, c].set_title(f't+{c + 1}', fontsize=8)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    cbar = fig.colorbar(im, ax=axes[2, 0], orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label('Absolute Error', fontsize=7)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_cloud_compact(seq_idx, gt, pred, filename, frames=None):
    if frames is None:
        frames = [0, 2, 4, 7, 9]
    ncols = len(frames)
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.2))

    for ci, ti in enumerate(frames):
        axes[0, ci].imshow(gt[ti].squeeze(), cmap='gray', vmin=0, vmax=1)
        im = axes[1, ci].imshow(pred[ti].squeeze(), cmap='gray', vmin=0, vmax=1)

    for r in range(nrows):
        axes[r, 0].set_ylabel(['Ground Truth', 'Predicted'][r], fontsize=9)
    for ci, ti in enumerate(frames):
        axes[0, ci].set_title(f't+{ti + 1}', fontsize=9)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_cloud_comparison(seq_idx, gt, pred_w, pred_n, filename, frames=None):
    if frames is None:
        frames = [0, 2, 4, 7, 9]
    ncols = len(frames)
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))

    for ci, ti in enumerate(frames):
        axes[0, ci].imshow(gt[ti].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, ci].imshow(pred_w[ti].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[2, ci].imshow(pred_n[ti].squeeze(), cmap='gray', vmin=0, vmax=1)

    for r in range(nrows):
        axes[r, 0].set_ylabel(['Ground Truth', 'With Velocity', 'No Velocity'][r], fontsize=9)
    for ci, ti in enumerate(frames):
        axes[0, ci].set_title(f't+{ti + 1}', fontsize=9)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def save_cloud_mse_curve(frame_mse_w, frame_mse_n, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    frames = np.arange(1, len(frame_mse_w) + 1)
    ax.plot(frames, frame_mse_w, 'o-', color='#2ca02c', linewidth=2, markersize=6, label='With Velocity')
    if frame_mse_n is not None:
        ax.plot(frames, frame_mse_n, 's-', color='#d62728', linewidth=2, markersize=6, label='Without Velocity')
        ax.fill_between(frames, frame_mse_n, frame_mse_w, alpha=0.15, color='#2ca02c')
    ax.set_xlabel('Prediction Timestep', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Per-Frame Prediction Error (GOES-16 Cloud Data)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(frames)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


print(f"Device: {device}")
print("Loading GOES-16 cloud test data...")
cloud_test = GOESCloud(
    root='data/cloud', date_str=CLOUD_DATE,
    start_hour=CLOUD_START, hours=CLOUD_HOURS, interval_min=CLOUD_INTERVAL,
    crop_size=64, n_frames_input=10, n_frames_output=10,
    is_train=False, train_ratio=0.8
)
test_loader = torch.utils.data.DataLoader(
    dataset=cloud_test, batch_size=1, shuffle=False, num_workers=0
)
print(f"  Test sequences: {len(cloud_test)}")

print("\nLoading velocity model...")
encoder_w = build_model()
encoder_w.load_state_dict(torch.load('save/encoder_cloud.pth', map_location=device))
encoder_w.eval()
vel_scale = encoder_w.phycell.cell_list[0].velocity_scale.item()
print(f"  Loaded. velocity_scale = {vel_scale:.4f}")

encoder_n = None
if DO_COMPARE:
    print("\nTraining no-velocity cloud model (50 epochs)...")
    cloud_train = GOESCloud(
        root='data/cloud', date_str=CLOUD_DATE,
        start_hour=CLOUD_START, hours=CLOUD_HOURS, interval_min=CLOUD_INTERVAL,
        crop_size=64, n_frames_input=10, n_frames_output=10,
        is_train=True, train_ratio=0.8
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=cloud_train, batch_size=4, shuffle=True, num_workers=0
    )
    encoder_n = build_model()
    encoder_n = train_cloud_no_velocity(encoder_n, train_loader, nepochs=50)
    encoder_n.eval()
    torch.save(encoder_n.state_dict(), 'save/encoder_cloud_no_velocity.pth')

print("\nEvaluating on test set...")
seq_errors = []
running_mse_w = np.zeros(PRED_FRAMES)
running_mse_n = np.zeros(PRED_FRAMES)
n_seqs = 0

t0 = time.time()
for i, out in enumerate(test_loader):
    inp = out[1].to(device)
    tgt = out[2].to(device)
    inv = out[3].to(device)
    outv = out[4].to(device)

    target_np = tgt.cpu().numpy()[0]
    preds_w = predict_cloud(encoder_w, inp, inv, outv, use_velocity=True)[0]
    mse_w = np.mean((preds_w - target_np) ** 2, axis=(1, 2, 3))

    if encoder_n is not None:
        preds_n = predict_cloud(encoder_n, inp, inv, outv, use_velocity=False)[0]
        mse_n = np.mean((preds_n - target_np) ** 2, axis=(1, 2, 3))
        running_mse_n += mse_n
    else:
        preds_n = None
        mse_n = np.zeros_like(mse_w)

    running_mse_w += mse_w
    n_seqs += 1

    seq_errors.append({
        'index': i,
        'mse_w_total': mse_w.mean(),
        'mse_n_total': mse_n.mean(),
        'gap': mse_n.mean() - mse_w.mean(),
        'mse_w_frame': mse_w,
        'mse_n_frame': mse_n,
        'input': inp.clone(),
        'target': tgt.clone(),
        'inv': inv.clone(),
        'outv': outv.clone(),
    })

elapsed = time.time() - t0
print(f"  Evaluated {n_seqs} sequences in {elapsed:.0f}s")

frame_mse_w = running_mse_w / n_seqs
frame_mse_n = running_mse_n / n_seqs if encoder_n is not None else None
avg_mse_w = frame_mse_w.mean()
avg_mse_n = frame_mse_n.mean() if frame_mse_n is not None else None

print(f"\n  With velocity:    avg MSE = {avg_mse_w:.4f}")
if avg_mse_n is not None:
    print(f"  Without velocity: avg MSE = {avg_mse_n:.4f}")
    print(f"  Improvement:      {(avg_mse_n - avg_mse_w):.4f} ({(avg_mse_n - avg_mse_w) / avg_mse_n * 100:.1f}%)")

print("\nGenerating WSI cloud figures...")

save_cloud_mse_curve(frame_mse_w, frame_mse_n, 'figures/per_frame_mse_cloud.png')

seq_errors.sort(key=lambda x: -x['gap'])
worst_gap = seq_errors[0]

for rank, item in enumerate(seq_errors[:2]):
    idx = item['index']
    gap = item['gap']

    print(f"\n  Worst #{rank + 1}: seq {idx}, gap={gap:.6f} (MSE w={item['mse_w_total']:.4f}, n={item['mse_n_total']:.4f})")

    inp_t = item['input'].to(device)
    tgt_t = item['target'].to(device)
    inv_t = item['inv'].to(device)
    outv_t = item['outv'].to(device)

    target_np = tgt_t.cpu().numpy()[0]
    preds_w = predict_cloud(encoder_w, inp_t, inv_t, outv_t, use_velocity=True)[0]
    diff_w = np.abs(preds_w - target_np)

    save_cloud_prediction_grid(
        idx, target_np, preds_w, diff_w,
        f'figures/cloud_worst_{rank + 1}_full.png', frames_to_show=10
    )
    save_cloud_compact(
        idx, target_np, preds_w,
        f'figures/cloud_worst_{rank + 1}_compact.png'
    )

    if encoder_n is not None:
        preds_n = predict_cloud(encoder_n, inp_t, inv_t, outv_t, use_velocity=False)[0]
        save_cloud_comparison(
            idx, target_np, preds_w, preds_n,
            f'figures/cloud_worst_{rank + 1}_comparison.png'
        )

print(f"\nDone. All cloud figures saved to figures/")
print(f"velocity_scale: {vel_scale:.4f}")
