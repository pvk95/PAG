import sys

sys.path.append('./data/')
from data_utils import *
import glob
import argparse
import os

import h5py

N_SAMPLES = 337

NEW_SHAPE = (232, 152)
MIN_HU = -1000
MAX_HU = 1000

ROOT = '/'
PATH_DATADIR = '/data/'
DAYS = [f'day{i}/' for i in np.arange(2, 31)]

MAP_TAGS = {'tumor-tag': 1, 'node-tag': 1, 'mets-tag': 1, 'other-tag': 0}


def pre_extra_images(ct, pet, lab):
    """
    Threshold CT image. extract the region of lungs and normalize the images to range of [0-1].
    """

    assert ct.shape[0] == 440 or ct.shape[0] == 485

    begin = 180
    height = 140 if ct.shape[0] == 440 else 160
    end = begin + height

    y_begin = 115
    y_end = 426

    x_begin = 32
    x_end = 496

    ct = pre_ct(ct)[begin:end, y_begin:y_end, x_begin:x_end]
    pet = pet[begin:end, y_begin:y_end, x_begin:x_end]
    lab = lab[begin:end, y_begin:y_end, x_begin:x_end]

    ct = normalize(ct).astype(np.float32)
    pet = normalize(pet).astype(np.float32)

    return ct, pet, lab


def process_extra_data(ct, pet, lab, label_names, data_idx, new_shape, map_tags):
    """
    Decide whether or not include the image as part of data based on set of criterion.
    """

    if ct.shape[0] == 440 or ct.shape[0] == 485:

        ct, pet, lab = pre_extra_images(ct, pet, lab)

        x_dim = ct.shape[1]
        temp_shape = (x_dim, 96)

        ct = opencv_resize(img=ct, new_shape=temp_shape)
        pet = opencv_resize(img=pet, new_shape=temp_shape)
        lab = opencv_resize(img=lab, new_shape=temp_shape)

        ct = opencv_resize(ct.transpose((1, 2, 0)), new_shape)
        pet = opencv_resize(pet.transpose((1, 2, 0)), new_shape)
        lab = opencv_resize(lab.transpose((1, 2, 0)), new_shape)

        try:
            lab, lab_points = modify_label(lab=lab, label_names=label_names, map_tags=map_tags)
        except ValueError:

            print("++++ Something strange! Skipping image +++++")
            return None
        except AssertionError:

            print("++++ Something strange! Skipping image +++++")
            return None

        print(f"Finalized image: {data_idx}")

        return ct, pet, lab, lab_points
    else:
        return None


def generate_data(save_dir):
    extra_images = np.zeros(shape=(100, 1, 152, 232, 96, 3), dtype=np.float32)

    examples = []
    for d in DAYS:
        path = PATH_DATADIR + d
        examples = examples + glob.glob(path + '*/')

    with open(os.path.join(save_dir, 'meta_data.pkl'), 'rb') as my_dict:
        meta_data = pickle.load(my_dict)
    with open(os.path.join(save_dir, 'index_data.pkl'), 'rb') as my_dict:
        index_data = pickle.load(my_dict)
    data_idx = len(index_data)

    extra_meta_data = {}
    extra_index_data = {}

    counter = 0
    for ex in examples:
        print(ex)

        if meta_data[ex]['data_idx'] is None:

            print(f"Counter : {counter}")

            d = np.load(ex + 'lsa.npz')
            pet = d['PET']
            ct = d['CT']
            lab = d['Labels']
            label_names = d['label_names']

            out = process_extra_data(ct, pet, lab, label_names, data_idx, NEW_SHAPE, MAP_TAGS)
            extra_meta_data[ex] = meta_data[ex]

            if out is not None:
                ct, pet, lab, lab_points = out

                extra_images[counter, 0, :, :, :, 0] = ct
                extra_images[counter, 0, :, :, :, 1] = pet
                extra_images[counter, 0, :, :, :, 2] = lab

                extra_meta_data[ex] = {'data_idx': data_idx, 'lab_points': lab_points}
                extra_index_data[ex] = data_idx

                counter += 1

                print("Processed image", data_idx)
                data_idx = data_idx + 1

    extra_images = extra_images[:counter, ...]

    with h5py.File(os.path.join(save_dir, 'extra_images.h5'), 'w') as file:
        file.create_dataset('data/data', data=extra_images, dtype=np.float32)

    with open(os.path.join(save_dir, 'extra_meta_data.pkl'), 'wb') as my_dict:
        pickle.dump(extra_meta_data, my_dict)

    with open(os.path.join(save_dir, 'extra_index_data.pkl'), 'wb') as my_dict:
        pickle.dump(extra_index_data, my_dict)


def generate_fun(data_file):
    writer = animation.FFMpegWriter(fps=30, codec='libx264')

    with h5py.File(data_file, 'r') as file:
        images = file['data/data'][()]

    for i, image in enumerate(images):
        print("Rendering {}.mp4".format(i))

        ct = np.squeeze(image[..., 0])
        pet = np.squeeze(image[..., 1])
        lab = np.squeeze(image[..., 2])

        ani = render(ct, pet, lab)

        mov_name = os.path.join(save_dir, 'animations/extra_{}.mp4'.format(i))

        ani.save(mov_name, writer=writer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Extra Data for test data')
    parser.add_argument('--save_dir', type=str, default='tumor_data/', help='Where to save data?')
    parser.add_argument('--gen', dest='gen', action='store_true')
    parser.add_argument('--anim', dest='anim', action='store_true')

    args = parser.parse_args()

    save_dir = os.path.join(ROOT, args.save_dir)

    if args.gen:
        generate_data(save_dir=save_dir)

    if args.anim:
        generate_fun(data_file=os.path.join(save_dir, 'extra_images.h5'))
