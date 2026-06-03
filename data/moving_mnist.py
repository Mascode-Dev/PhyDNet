import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        vel_y = np.zeros(seq_length)
        vel_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x
            vel_x[i] = v_x * self.step_length_
            vel_y[i] = v_y * self.step_length_

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x, vel_y, vel_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        all_vel_x = np.zeros((num_digits, self.n_frames_total), dtype=np.float32)
        all_vel_y = np.zeros((num_digits, self.n_frames_total), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x, vel_y, vel_x = self.get_random_trajectory(self.n_frames_total)
            all_vel_x[n] = vel_x
            all_vel_y[n] = vel_y
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        # Aggregate velocity: mean across all digits per frame
        mean_vel_x = all_vel_x.mean(axis=0)  # (n_frames_total,)
        mean_vel_y = all_vel_y.mean(axis=0)  # (n_frames_total,)
        velocities = np.stack([mean_vel_x, mean_vel_y], axis=-1)  # (n_frames_total, 2)

        data = data[..., np.newaxis]
        return data, velocities

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images, velocities = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]
            # Estimate velocity from frame differencing (center-of-mass)
            velocities = self._estimate_velocities(images.squeeze(-1))  # (n_frames_total, 2)

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1 # patch size (a 4 dans les PredRNN)
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        input_velocity = velocities[:self.n_frames_input]
        if self.n_frames_output > 0:
            output_velocity = velocities[self.n_frames_input:length]
        else:
            output_velocity = []

        frozen = input[-1]

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        input_velocity = torch.from_numpy(input_velocity).contiguous().float()
        output_velocity = torch.from_numpy(output_velocity).contiguous().float()

        out = [idx, input, output, input_velocity, output_velocity]
        return out

    def _estimate_velocities(self, frames):
        """Estimate velocity from center-of-mass frame differencing for test set."""
        n_frames = frames.shape[0]
        velocities = np.zeros((n_frames, 2), dtype=np.float32)
        yy, xx = np.mgrid[0:frames.shape[1], 0:frames.shape[2]]
        for i in range(1, n_frames):
            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            prev_sum = prev.sum()
            curr_sum = curr.sum()
            if prev_sum > 0 and curr_sum > 0:
                prev_cx = (xx * prev).sum() / prev_sum
                prev_cy = (yy * prev).sum() / prev_sum
                curr_cx = (xx * curr).sum() / curr_sum
                curr_cy = (yy * curr).sum() / curr_sum
                velocities[i, 0] = (curr_cx - prev_cx) / self.image_size_
                velocities[i, 1] = (curr_cy - prev_cy) / self.image_size_
        # Frame 0 velocity = frame 1 velocity
        if n_frames > 1:
            velocities[0] = velocities[1]
        return velocities

    def __len__(self):
        return self.length
