"""
Author: Varaha Karthik

The file that implements the data pre-processing pipeline.
"""
import sys

sys.path.append('./data/')
from data_utils import *
import glob
import argparse
import os

import albumentations as alb
from albumentations import (HorizontalFlip, VerticalFlip, Flip,
                            RandomRotate90, Transpose, Compose, OneOf,
                            ElasticTransform, GridDistortion, OpticalDistortion)

import random
import time
import h5py

N_SAMPLES = 337  # Total number of samples. I already know the number of samples from before.

NEW_SHAPE = (232, 152)  # Resize to this shape (y,x)
MIN_HU = -1000  # Minimum HU for CT
MAX_HU = 1000  # Maximum HU for CT

ROOT = '/home/karthikp/MST/'
PATH_DATADIR = '/data/'  # The place where the data resides
DAYS = [f'day{i}/' for i in np.arange(2, 31)]  # Sub-folder to where the data resides.

# All the tumors are considered in one single class. Change accordingly for dfferent classes.
MAP_TAGS = {'tumor-tag': 1, 'node-tag': 1, 'mets-tag': 1, 'other-tag': 0}


def animate(ct, pet, lab, writer, data_idx, save_dir):
    """
    Create a movie of the image. Appropriate if you want to view the image.
    :param ct: CT image
    :param pet: PET images
    :param lab: Labeled image
    :param writer: The movie writer
    :param data_idx: The particular index of the data
    :param save_dir: Where to save the animations?
    :return:
    """
    ani = render(np.squeeze(ct), np.squeeze(pet), np.squeeze(lab))
    mov_name = os.path.join(save_dir, f'animations/mov_{data_idx}.mp4')
    ani.save(mov_name, writer=writer)


def generate_data(save_dir, anim=False):
    """
    Main function that generates the data
    :param save_dir: Where to save data
    :param anim: Whether to generate animations
    :return:
    """

    # Initialize a numpy array with zeros and then load the preprocessed data. Optimal memory usage.
    init_data(save_dir=save_dir, n_samples=N_SAMPLES)

    # Load all the folders where the data resides. And then sequentially load the images.
    examples = []
    for d in DAYS:
        path = PATH_DATADIR + d
        examples = examples + glob.glob(path + '*/')

    meta_data = {}  # Meta data that contains all the information of the data. (Except the labels)
    index_data = {}  # Points the index to appropriate data indentifier.
    data_idx = 0

    writer = animation.FFMpegWriter(fps=30, codec='libx264')

    img_counter = 1
    for ex in examples:

        print("======== Image counter: ========", img_counter)
        img_counter = img_counter + 1
        print(ex)
        # Load the .npy file. It contains the images.
        d = np.load(ex + 'lsa.npz')
        pet = d['PET']
        ct = d['CT']
        lab = d['Labels']
        spacing = d['spacing']
        label_names = d['label_names']

        try:
            clinical_staging = d['clinical_staging']
        except KeyError:
            clinical_staging = None
            print("Clinical staging info not found")
        except ValueError:
            clinical_staging = None
            print("Error loading clincial staging")

        meta_data[ex] = {"shape": ct.shape, "spacing": spacing, "label_names": label_names,
                         "clinical_staging": clinical_staging,
                         "data_idx": None, "lab_points": None}

        # Pre-process the data
        out = process_data(ct, pet, lab, label_names, data_idx, NEW_SHAPE, MAP_TAGS)

        if out is not None:
            ct, pet, lab, lab_points = out

            meta_data[ex]["data_idx"] = data_idx
            meta_data[ex]["lab_points"] = lab_points

            index_data[data_idx] = ex

            flush_data(ct, pet, lab, meta_data, index_data, data_idx, save_dir)

            if anim:
                animate(ct=ct, pet=pet, lab=lab, writer=writer, data_idx=data_idx, save_dir=save_dir)

            data_idx = data_idx + 1
            print("Processed image")


# ===================================
# ========== Augmentation ===========
# ===================================

def augment_data(save_dir):
    """
    A special that implemnets the data augmentation pipeline.
    :param save_dir: Where to save the augmented data?
    :return:
    """

    seed = 1337
    random.seed(seed)
    start_time = time.time()
    print(f"====== Augmenting data. Seed set at {seed} ======")

    data_file = h5py.File(os.path.join(save_dir, 'data_file.h5'), 'r')
    data_shape = data_file['data/data'].shape

    data_aug = np.zeros(shape=data_shape, dtype=np.float32)

    n_samples = data_shape[0]
    img_channels, img_height, img_width, img_depth = data_shape[1:5]

    try:
        aug = alb.load(os.path.join(save_dir, 'aug_pipeline_1.json'))
    except FileNotFoundError:
        print("Pipeline not found. Generating One ...")
        aug = Compose([
            OneOf([VerticalFlip(p=1), HorizontalFlip(p=1)], p=1),
            OneOf([ElasticTransform(p=1, sigma=6, alpha_affine=4, alpha=75),
                   GridDistortion(p=1),
                   OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)], p=0.8)])

        alb.save(aug, os.path.join(save_dir, 'aug_pipeline_1.json'))

    for data_idx in np.arange(n_samples):
        img = data_file['data/data'][data_idx, ...]
        img = img.reshape(img_channels, img_height, img_width, -1)
        img_aug = aug(image=img[0, ...])['image'].reshape(img_channels, img_height, img_width, img_depth, -1)

        data_aug[data_idx, ...] = img_aug

        del img_aug
        del img

    data_file.close()

    with h5py.File(os.path.join(save_dir, 'data_aug.h5'), 'w') as file:
        file.create_dataset('data/data', data=data_aug, dtype=np.float32)

    print(f"====== Finished augmentation. Time taken: {time.time() - start_time}s ======")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Data pre-processing for PAG model')
    parser.add_argument('--anim', dest='anim', action='store_true', help='Generate animations of pre-processed images?')
    parser.add_argument('--save_dir', type=str, default='tumor_data/', help='Where to save data?')
    parser.add_argument('--data_aug', dest='data_aug', action='store_true', help='Should perform data augmentation?')

    args = parser.parse_args()

    anim = args.anim
    save_dir = os.path.join(ROOT, args.save_dir)

    if anim:
        try:
            os.makedirs(os.path.join(save_dir, 'animations'))
        except FileExistsError:
            pass

    generate_data(anim=anim, save_dir=save_dir)

    if args.data_aug:
        augment_data(save_dir=save_dir)
