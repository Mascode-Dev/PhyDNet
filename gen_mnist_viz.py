"""Load saved checkpoints from compare_velocity.py and generate prediction visualizations."""

import torch
import torch.nn as nn
import numpy as np
import os
from models.models import ConvLSTM, PhyCell, EncoderRNN
from data.moving_mnist import MovingMNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NUM_SEQUENCES = 2
PRED_FRAMES = 5

mm_test = MovingMNIST(root='data/', is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mm_test, batch_size=NUM_SEQUENCES, shuffle=False, num_workers=0)


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


def create_viz_grid(frames_list, row_titles, filename, ncols):
    nrows = len(frames_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(frames_list[r][c].squeeze(), cmap='gray', vmin=0, vmax=1)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9)
            if r == 0:
                ax.set_title(f't+{c+1}', fontsize=9)
            ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def build_model():
    phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                      n_layers=1, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                        n_layers=3, kernel_size=(3, 3), device=device)
    encoder = EncoderRNN(phycell, convcell, device)
    return encoder


os.makedirs('save', exist_ok=True)

# Load checkpoints
print("Loading checkpoints...")
encoder_no_vel = build_model()
encoder_no_vel.load_state_dict(torch.load('save/encoder_no_velocity_e99.pth', map_location=device))
encoder_no_vel.eval()

encoder_with_vel = build_model()
encoder_with_vel.load_state_dict(torch.load('save/encoder_with_velocity_e99.pth', map_location=device))
encoder_with_vel.eval()
print("  Loaded both checkpoints")

# Get test sequences
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
    gt_seq = [target[seq_idx, f] for f in range(PRED_FRAMES)]
    pw_seq = [preds_with[seq_idx, f] for f in range(PRED_FRAMES)]
    pn_seq = [preds_no[seq_idx, f] for f in range(PRED_FRAMES)]

    create_viz_grid(
        [gt_seq, pw_seq],
        ['Ground Truth', 'With Velocity'],
        f'figures/mnist_seq{seq_idx+1}_predictions.png',
        PRED_FRAMES
    )

    create_viz_grid(
        [gt_seq, pw_seq, pn_seq],
        ['Ground Truth', 'With Velocity', 'No Velocity'],
        f'figures/mnist_seq{seq_idx+1}_comparison.png',
        PRED_FRAMES
    )

print("\nDone. Figures saved to figures/")
