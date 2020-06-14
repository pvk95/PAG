import numpy as np
import h5py
import pickle
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# ================================================================================================ #
# ======================== Operations on CT, PET and Labelled Images ============================= #
# ================================================================================================ #

def normalize(img):
    """
    Normalize the input image to [0-1] range.
    :param img: [height, width, depth] order does not matter as long it is a 3d image,
    :return: Normalized image
    """
    max_new = 1.0
    min_new = 0.0
    img_max = img.max()
    img_min = img.min()

    return ((img - img_min) * (max_new - min_new)) / (img_max - img_min) + min_new


def pre_ct(ct, min_hu=-1000, max_hu=1000):
    """
    Threshold the CT image so that anything above MAX_HU is reduced to MAX_HU and anything below MIN_HU is reduced to
    MIN_HU.
    """
    temp = ct.copy()

    temp[temp < min_hu] = min_hu
    temp[temp > max_hu] = max_hu

    return temp.astype(np.float32)


def pre_images(ct, pet, lab):
    """
    Threshold CT image. extract the region of lungs and normalize the images to range of [0-1].
    """
    begin = None
    end = None

    if ct.shape[0] < 300:
        begin = 100
        end = 196
    elif 300 < ct.shape[0] < 350:
        begin = 130
        end = 226

    y_begin = 120
    y_end = 416

    x_begin = 32
    x_end = 496

    ct = pre_ct(ct)[begin:end, y_begin:y_end, x_begin:x_end]
    pet = pet[begin:end, y_begin:y_end, x_begin:x_end]
    lab = lab[begin:end, y_begin:y_end, x_begin:x_end]

    ct = normalize(ct).astype(np.float32)
    pet = normalize(pet).astype(np.float32)

    return ct, pet, lab


def opencv_resize(img, new_shape):
    """
    Resize the image for every axial slice
    :param img: [L, W, H]
    :param new_shape: Resize to new_shape
    :return:
    """

    z_dim = img.shape[2]
    new_img = np.stack([cv2.resize(img[:, :, i], dsize=new_shape,
                                   interpolation=cv2.INTER_NEAREST) for i in range(z_dim)], axis=-1)

    return new_img


# ================================================================================================ #
# ================= Modifications to Annotations depending on label_names ======================== #
# ================================================================================================ #


def lab_info(label):
    """
    Extract the tumor stage, comment/location, the labeling index and the type of tumor
    :param label: The label
    :return:
    """
    out = label.decode('utf-8').split('=')
    before_eq = out[0]
    after_eq = out[1]

    before_out = before_eq.split('_')
    label_index = int(before_out[0]) + 1

    # print("++++ Something strange! Skipping image +++++")
    # print(label)
    # label_index = int(input("Enter the label: ")) + 1

    tag = before_out[1]

    after_out = after_eq.partition('_')
    stage = after_out[0]
    comment = after_out[2]

    return label_index, tag, stage, comment


def get_change_labels(label_names, map_tags):
    """
    Merge all the labels of a particular tumor type
    :param label_names: The names of the labels
    :param map_tags: What is final label each type of tumor takes?
    :return:
    """
    change_label = {}

    for name in label_names:
        label_index, tag, stage, comment = lab_info(name)

        assert label_index not in change_label.keys()
        change_label[label_index] = map_tags[tag]

    return change_label


def modify_label(lab, label_names, map_tags):
    """
    The API for merging labels
    :param lab:
    :param label_names: Label names
    :param map_tags:
    :return:
    """
    change_label = get_change_labels(label_names=label_names, map_tags=map_tags)

    lab_modify = np.zeros_like(lab).astype(np.int)

    lab_points = {}
    label_idxs = list(change_label.keys())

    print("Label names: ", label_names)

    for lb in label_idxs:
        assert lb not in lab_points.keys()
        lab_points[lb] = np.array(list(zip(*np.where(lab == lb))))
        try:
            assert lab_points[lb].shape[0] > 0
            for p in lab_points[lb]:
                lab_modify[p[0], p[1], p[2]] = change_label[lb]
        except AssertionError:
            print(f"{lb} not found.")

    return lab_modify, lab_points


def process_data(ct, pet, lab, label_names, data_idx, new_shape, map_tags):
    """
    Decide whether or not include the image as part of data based on set of criterion.
    """

    cond = ct.shape[0] > 350 or ct.shape[0] < 250

    if cond:
        return None

    ct, pet, lab = pre_images(ct, pet, lab)

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


def init_data(save_dir, n_samples):
    shape = (1, 152, 232, 96, 3)
    zeros_data = np.zeros(shape=(n_samples,) + shape, dtype=np.float32)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with h5py.File(save_dir + 'data_file.h5', 'w') as file:
        file.create_group(name='data')
        file.create_dataset(name='data/data', data=zeros_data, dtype=np.float32)


def flush_data(ct, pet, lab, meta_data, index_data, data_index, save_dir):
    with h5py.File(save_dir + 'data_file.h5', 'a') as file:
        img = np.stack((ct, pet, lab), axis=-1)
        file['data/data'][data_index, ...] = img

    with open(save_dir + 'meta_data.pkl', 'wb') as my_dict:
        pickle.dump(meta_data, my_dict)

    with open(save_dir + 'index_data.pkl', 'wb') as my_dict:
        pickle.dump(index_data, my_dict)


# ================================================================================================ #
# ================= Modifications to Annotations depending on label_names ======================== #
# ================================================================================================ #

def alpha_cmap():
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)][::-1]
    # force the first color entry to be grey
    cmaplist[0] = (0, 0, 0, 1)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.arange(6)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def render(ct, pet, lab):
    def animate(i):
        # CT
        ax1.set_data(ct[:, :, i])

        # CT + PET
        ax2.set_data(ct[:, :, i])
        ax2_2.set_data(pet[:, :, i])

        # CT + Segmentation
        ax3.set_data(ct[:, :, i])
        ax3_3.set_data(lab[:, :, i])

        return ax1, ax2, ax3,

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=200)
    axes[0].set_title("CT")
    ax1 = axes[0].imshow(ct[..., 0], cmap=plt.cm.gray)

    axes[1].set_title("CT + PET")
    ax2 = axes[1].imshow(ct[..., 0], cmap=plt.cm.gray)
    ax2_2 = axes[1].imshow(pet[..., 0], cmap=plt.cm.gist_heat, alpha=0.5)

    axes[2].set_title("CT + Segmentation")
    ax3 = axes[2].imshow(ct[..., 0], cmap=plt.cm.gray)

    cmap, norm = alpha_cmap()
    ax3_3 = axes[2].imshow(lab[..., 0], cmap=cmap, norm=norm, alpha=0.5)

    ani = animation.FuncAnimation(fig, animate, frames=ct.shape[-1], interval=100, repeat_delay=1000)

    plt.close()

    return ani
