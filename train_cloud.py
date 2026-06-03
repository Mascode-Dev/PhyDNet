"""Train PhyDNet on GOES-16 cloud satellite imagery and visualize predictions."""

import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from models.models import ConvLSTM, PhyCell, EncoderRNN
from data.goes_cloud import GOESCloud
from constrain_moments import K2M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NEPOCHS = 50
EVAL_EVERY = 10
BATCH_SIZE = 4
PRED_FRAMES = 5
NUM_SEQUENCES = 2

GOES_DATE = '2024-07-14'
GOES_START_HOUR = 18
GOES_HOURS = 6
GOES_INTERVAL = 5

os.makedirs('save', exist_ok=True)

print("Loading GOES-16 cloud data...")
train_set = GOESCloud(
    root='data/cloud', date_str=GOES_DATE,
    start_hour=GOES_START_HOUR, hours=GOES_HOURS, interval_min=GOES_INTERVAL,
    crop_size=64, n_frames_input=10, n_frames_output=10,
    is_train=True, train_ratio=0.8
)
test_set = GOESCloud(
    root='data/cloud', date_str=GOES_DATE,
    start_hour=GOES_START_HOUR, hours=GOES_HOURS, interval_min=GOES_INTERVAL,
    crop_size=64, n_frames_input=10, n_frames_output=10,
    is_train=False, train_ratio=0.8
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=NUM_SEQUENCES, shuffle=False, num_workers=0
)

print(f"Train sequences: {len(train_set)}, Test sequences: {len(test_set)}")

constraints = torch.zeros((49, 7, 7)).to(device)
ind = 0
for i in range(0, 7):
    for j in range(0, 7):
        constraints[ind, i, j] = 1
        ind += 1


def train_on_batch(input_tensor, target_tensor, input_velocity, output_velocity,
                   encoder, optimizer, criterion, teacher_forcing_ratio):
    optimizer.zero_grad()
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0

    for ei in range(input_length - 1):
        _, _, output_image, _, _ = encoder(
            input_tensor[:, ei, :, :, :], (ei == 0),
            velocity=input_velocity[:, ei, :]
        )
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]
    use_tf = random.random() < teacher_forcing_ratio

    for di in range(target_length):
        _, _, output_image, _, _ = encoder(
            decoder_input,
            velocity=output_velocity[:, di, :]
        )
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


def predict_sequence(encoder, input_tensor, input_velocity, output_velocity):
    with torch.no_grad():
        input_length = input_tensor.size(1)
        target_length = output_velocity.size(1)

        for ei in range(input_length - 1):
            encoder(input_tensor[:, ei, :, :, :], (ei == 0),
                    velocity=input_velocity[:, ei, :])

        decoder_input = input_tensor[:, -1, :, :, :]
        predictions = []
        for di in range(target_length):
            _, _, output_image, _, _ = encoder(
                decoder_input, velocity=output_velocity[:, di, :]
            )
            decoder_input = output_image
            predictions.append(output_image.cpu().numpy())

        return np.stack(predictions).swapaxes(0, 1)


def save_viz_grid(gt_frames, pred_frames, filename):
    fig, axes = plt.subplots(2, PRED_FRAMES, figsize=(PRED_FRAMES * 2.2, 4.5))
    for r in range(2):
        for c in range(PRED_FRAMES):
            ax = axes[r, c]
            ax.imshow(pred_frames[r][c].squeeze(), cmap='gray', vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f'GT t+{c+1}', fontsize=9)
            else:
                ax.set_title(f'Pred t+{c+1}', fontsize=9)
            ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


print("\nBuilding model...")
phycell = PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49],
                  n_layers=1, kernel_size=(7, 7), device=device)
convcell = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64],
                    n_layers=3, kernel_size=(3, 3), device=device)
encoder = EncoderRNN(phycell, convcell, device)

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
criterion = nn.MSELoss()

print(f"Training {NEPOCHS} epochs...")
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
                              encoder, optimizer, criterion, tf_ratio)
        loss_epoch += loss

    epoch_time = time.time() - t0
    print(f"  epoch {epoch:3d}  loss {loss_epoch:.2f}  time {epoch_time:.0f}s  tf {tf_ratio:.3f}")

    if (epoch + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            test_loss = 0
            for out in test_loader:
                inp = out[1].to(device)
                tgt = out[2].to(device)
                inv = out[3].to(device)
                outv = out[4].to(device)
                preds = predict_sequence(encoder, inp, inv, outv)
                preds_t = torch.from_numpy(preds).to(device)
                test_loss += criterion(preds_t, tgt).item()
            test_loss /= len(test_loader)
            print(f"  >>> EVAL  Test MSE: {test_loss:.6f}")
            scheduler.step(test_loss)

    if (epoch + 1) % 50 == 0:
        torch.save(encoder.state_dict(), f'save/encoder_cloud_e{epoch+1}.pth')

total_time = time.time() - total_t0
print(f"\nTotal training time: {total_time/3600:.1f}h")

torch.save(encoder.state_dict(), 'save/encoder_cloud.pth')
vel_scale = encoder.phycell.cell_list[0].velocity_scale.item()
print(f"Learned velocity_scale: {vel_scale:.4f}")

print("\nGenerating cloud prediction visualizations...")
test_iter = iter(test_loader)
out = next(test_iter)
input_tensor = out[1].to(device)
target_tensor = out[2].to(device)
input_velocity = out[3].to(device)
output_velocity = out[4].to(device)

target = target_tensor[:, :PRED_FRAMES].cpu().numpy()
preds = predict_sequence(encoder, input_tensor, input_velocity, output_velocity)
preds = preds[:, :PRED_FRAMES]

for seq_idx in range(min(NUM_SEQUENCES, target.shape[0])):
    gt_seq = [target[seq_idx, f] for f in range(PRED_FRAMES)]
    pr_seq = [preds[seq_idx, f] for f in range(PRED_FRAMES)]
    save_viz_grid(
        gt_frames=[gt_seq, pr_seq],
        pred_frames=[gt_seq, pr_seq],
        filename=f'figures/cloud_seq{seq_idx+1}_predictions.png'
    )

print("\nDone. Figures saved to figures/")
