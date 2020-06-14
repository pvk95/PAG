import h5py
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import sys
sys.path.append('./validate/')
from analysis_utils import train_valid_test_idxs


def render(img, pet_recon, mov_name='mov.mp4'):
    def animate(i):
        ax1.set_data(img[0, :, :, i, 0])
        ax1_1.set_data(img[0, :, :, i, 2])

        ax2.set_data(img[0, :, :, i, 1])

        ax3.set_data(pet_recon[0, :, :, i])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=100)

    axes = axes.ravel()

    for axz in axes:
        axz.axis('off')

    axes[0].set_title('CT')
    ax1 = axes[0].imshow(img[0, :, :, 0, 0], cmap=plt.cm.gray)
    ax1_1 = axes[0].imshow(img[0, :, :, 0, 0], cmap=plt.cm.gray, alpha=0.5)

    axes[1].set_title('PET')
    ax2 = axes[1].imshow(img[0, :, :, 0, 1], cmap='gist_heat', vmin=0, vmax=1)

    axes[2].set_title('PET Recon')
    ax3 = axes[2].imshow(pet_recon[0, :, :, 0], cmap='gist_heat', vmin=0, vmax=1)

    fig.colorbar(ax2, ax=[axes[1], axes[2]], location='bottom')

    writer = animation.FFMpegWriter(fps=30, codec='libx264')
    ani = animation.FuncAnimation(fig, func=animate, frames=np.arange(1, img.shape[3]),
                                  interval=100, repeat_delay=1000)
    ani.save(mov_name, writer=writer)

    plt.close()

    return ani


def recon_analyze(data_type, save_dir):

    if not os.path.exists(os.path.join(save_dir, data_type)):
        os.makedirs(os.path.join(save_dir, data_type))

    train_idxs, valid_idxs, test_idxs = train_valid_test_idxs(save_dir=save_dir)

    if data_type == 'valid':
        data_indexes = valid_idxs
    elif data_type == 'train':
        data_indexes = train_idxs
    else:
        raise ValueError

    with h5py.File(SAVE_DIR + 'tumor_data/data_file.h5', 'r') as file:
        data = file['data/data'][data_indexes, ...]

    with h5py.File(save_dir + 'predictions.h5', 'r') as file:
        pet_data = file[f'pet/{data_type}'][()]

    n_samples = len(pet_data)

    for img_index in range(n_samples):

        data_idx = data_indexes[img_index]
        print("Rendering image: {}".format(data_idx))
        img = data[img_index, ...]
        pet_recon = pet_data[img_index, ...]
        mov_name = os.path.join(save_dir, data_type, 'recon_{}.mp4'.format(data_idx))
        render(img, pet_recon, mov_name)


if __name__ == '__main__':

    SAVE_DIR = '/'
    ROOT = '/'

    parser = argparse.ArgumentParser('PET Recon')
    parser.add_argument('--dirs', nargs='+', default='experiments/pet/cv0/')
    parser.add_argument('--valid', dest='valid', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    args = parser.parse_args()

    directories = args.dirs

    if args.valid:
        data_type = 'valid'
    elif args.train:
        data_type = 'train'
    else:
        print("Something should be input !")
        raise ValueError

    for d in directories:
        save_dir = os.path.join(ROOT, d)
        print(save_dir)
        recon_analyze(data_type=data_type, save_dir=save_dir)
