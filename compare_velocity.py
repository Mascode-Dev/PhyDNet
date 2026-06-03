"""Comparison script: PhyDNet with velocity vs without velocity.
Trains 100 epochs each, then visualizes predictions vs ground truth."""

import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from models.models import ConvLSTM, PhyCell, EncoderRNN
from data.moving_mnist import MovingMNIST
from constrain_moments import K2M
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NEPOCHS = 100
EVAL_EVERY = 20
BATCH_SIZE = 16
NUM_SEQUENCES = 2
PRED_FRAMES = 5

os.makedirs('save', exist_ok=True)

mm_train = MovingMNIST(root='data/', is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
train_loader = torch.utils.data.DataLoader(dataset=mm_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

mm_test = MovingMNIST(root='data/', is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mm_test, batch_size=NUM_SEQUENCES, shuffle=False, num_workers=0)

constraints = torch.zeros((49, 7, 7)).to(device)
ind = 0
for i in range(0, 7):
    for j in range(0, 7):
        constraints[ind, i, j] = 1
        ind += 1


def train_on_batch(input_tensor, target_tensor, input_velocity, output_velocity,
                   encoder, optimizer, criterion, teacher_forcing_ratio, use_velocity=True):
    optimizer.zero_grad()
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0

    for ei in range(input_length - 1):
        vel = input_velocity[:, ei, :] if use_velocity else None
        _, _, output_image, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0), velocity=vel)
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]
    use_tf = random.random() < teacher_forcing_ratio
    for di in range(target_length):
        vel = output_velocity[:, di, :] if use_velocity else None
        _, _, output_image, _, _ = encoder(decoder_input, velocity=vel)
        loss += criterion(output_image, target_tensor[:, di, :, :, :])
        decoder_input = target_tensor[:, di, :, :, :] if use_tf else output_image

    k2m = K2M([7, 7]).to(device)
    for b in range(encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
        m = k2m(filters.double()).float()
        loss += criterion(m, constraints)

    loss.backward()
    optimizer.step()
    return loss.item() / target_length


def evaluate(encoder, loader, use_velocity=True):
    total_mse, total_ssim = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for out in loader:
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_velocity = out[3].to(device)
            output_velocity = out[4].to(device)
            input_length = input_tensor.size(1)
            target_length = target_tensor.size(1)

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

            predictions = np.stack(predictions).swapaxes(0, 1)
            target = target_tensor.cpu().numpy()
            total_mse += np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()

            for a in range(target.shape[0]):
                for b in range(target.shape[1]):
                    total_ssim += ssim(target[a, b, 0], predictions[a, b, 0], data_range=1.0) / (target.shape[0] * target.shape[1])

    n = len(loader)
    elapsed = time.time() - t0
    return total_mse / n, total_ssim / n, elapsed


def predict_sequence(encoder, input_tensor, input_velocity, output_velocity, use_velocity=True):
    """Run autoregressive prediction for one batch. Returns predictions [(batch, 1, 64, 64), ...]."""
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


def build_model(use_velocity):
    phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                      n_layers=1, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                        n_layers=3, kernel_size=(3, 3), device=device)
    encoder = EncoderRNN(phycell, convcell, device)
    label = "with_velocity" if use_velocity else "no_velocity"
    return encoder, label


def run_experiment(use_velocity, label):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {label}")
    print(f"{'='*60}")

    phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                      n_layers=1, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                        n_layers=3, kernel_size=(3, 3), device=device)
    encoder = EncoderRNN(phycell, convcell, device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_mse = float('inf')
    total_t0 = time.time()

    for epoch in range(NEPOCHS):
        t0 = time.time()
        loss_epoch = 0
        tf_ratio = max(0, 1 - epoch * 0.003)

        for out in train_loader:
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_velocity = out[3].to(device)
            output_velocity = out[4].to(device)
            loss = train_on_batch(input_tensor, target_tensor, input_velocity, output_velocity,
                                  encoder, optimizer, criterion, tf_ratio, use_velocity=use_velocity)
            loss_epoch += loss

        epoch_time = time.time() - t0
        print(f"  epoch {epoch:3d}  loss {loss_epoch:8.2f}  time {epoch_time:.0f}s")

        if (epoch + 1) % EVAL_EVERY == 0:
            mse, ssim_val, eval_time = evaluate(encoder, test_loader, use_velocity=use_velocity)
            print(f"  >>> EVAL  MSE {mse:.4f}  SSIM {ssim_val:.4f}  ({eval_time:.0f}s)")
            if mse < best_mse:
                best_mse = mse

    total_time = time.time() - total_t0
    print(f"\n  Total time: {total_time/3600:.1f}h  Best MSE: {best_mse:.4f}")

    torch.save(encoder.state_dict(), f'save/encoder_{label}.pth')
    return encoder


def create_viz_grid(gt_frames, pred_frames, titles, filename, nrows, ncols):
    """Create a grid of images: ground truth row(s) + prediction row(s)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(pred_frames[r][c].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles[r][c], fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def visualize_predictions(encoder_with_vel, encoder_no_vel, test_loader):
    """Generate prediction visualizations for Moving MNIST."""
    test_iter = iter(test_loader)
    out = next(test_iter)
    input_tensor = out[1].to(device)
    target_tensor = out[2].to(device)
    input_velocity = out[3].to(device)
    output_velocity = out[4].to(device)

    target = target_tensor[:, :PRED_FRAMES].cpu().numpy()

    preds_with = predict_sequence(encoder_with_vel, input_tensor, input_velocity, output_velocity, use_velocity=True)
    preds_with = preds_with[:, :PRED_FRAMES]

    preds_no = predict_sequence(encoder_no_vel, input_tensor, input_velocity, output_velocity, use_velocity=False)
    preds_no = preds_no[:, :PRED_FRAMES]

    for seq_idx in range(NUM_SEQUENCES):
        # Figure 1: Ground truth vs with-velocity predictions
        gt_seq = [target[seq_idx, f] for f in range(PRED_FRAMES)]
        pw_seq = [preds_with[seq_idx, f] for f in range(PRED_FRAMES)]
        create_viz_grid(
            gt_frames=[gt_seq, pw_seq],
            pred_frames=[gt_seq, pw_seq],
            titles=[[f'GT t+{i+1}' for i in range(PRED_FRAMES)],
                    [f'Pred t+{i+1}' for i in range(PRED_FRAMES)]],
            filename=f'figures/mnist_seq{seq_idx+1}_predictions.png',
            nrows=2, ncols=PRED_FRAMES
        )

        # Figure 2: Ground truth vs with-velocity vs without-velocity
        pn_seq = [preds_no[seq_idx, f] for f in range(PRED_FRAMES)]
        create_viz_grid(
            gt_frames=[gt_seq, pw_seq, pn_seq],
            pred_frames=[gt_seq, pw_seq, pn_seq],
            titles=[[f'GT t+{i+1}' for i in range(PRED_FRAMES)],
                    [f'With vel t+{i+1}' for i in range(PRED_FRAMES)],
                    [f'No vel t+{i+1}' for i in range(PRED_FRAMES)]],
            filename=f'figures/mnist_seq{seq_idx+1}_comparison.png',
            nrows=3, ncols=PRED_FRAMES
        )


print("\nTraining both models (this will take a while)...")
encoder_no_vel = run_experiment(use_velocity=False, label="no_velocity")
encoder_with_vel = run_experiment(use_velocity=True, label="with_velocity")

visualize_predictions(encoder_with_vel, encoder_no_vel, test_loader)
print("\nDone. Figures saved to figures/")
