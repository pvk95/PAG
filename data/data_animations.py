import sys

sys.path.append('./data/')
from data_utils import *
import argparse
import multiprocessing as mp
import h5py


def generate_fun(samples, data_file, lock):
    for data_idx in samples:
        mov_name = os.path.join(save_dir, f'mov_{data_idx}.mp4')
        if not os.path.isfile(mov_name):

            print("Image: {}".format(data_idx))
            lock.acquire()
            with h5py.File(data_file, 'r') as file:
                image = file['data/data'][data_idx, ...]
            lock.release()

            ct = np.squeeze(image[..., 0])
            pet = np.squeeze(image[..., 1])
            lab = np.squeeze(image[..., 2])

            ani = render(ct, pet, lab)
            ani.save(mov_name, writer=writer)
        else:
            print("Animation for image: {} exists".format(data_idx))


if __name__ == '__main__':

    ROOT = '/home/karthikp/MST/'

    parser = argparse.ArgumentParser('Generate Animations for already generated data')
    parser.add_argument('--data_path', default='data/', help='Path of data whose animations you want')
    parser.add_argument('--save_dir', default='animations/', help='Where do you want to save animations.')
    parser.add_argument('--n_proc', default=2, help='No. of multi-processing', type=int)

    args = parser.parse_args()

    data_path = args.data_path
    save_dir = os.path.join(ROOT, data_path, args.save_dir)
    n_proc = args.n_proc

    try:

        data_file = os.path.join(ROOT, data_path, 'data_file.h5')

        with open(os.path.join(ROOT, data_path, 'meta_data.pkl'), 'rb') as my_dict:
            meta_data = pickle.load(my_dict)

        with open(os.path.join(ROOT, data_path, 'index_data.pkl'), 'rb') as my_dict:
            index_data = pickle.load(my_dict)

    except FileNotFoundError:
        print("Data files not found in path {}".format(data_path))
        sys.exit(1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = animation.FFMpegWriter(fps=30, codec='libx264')

    n_samples = len(index_data.keys())
    samples = np.arange(n_samples)
    n_proc_smp = n_samples // n_proc
    proc_samples = []

    for i in np.arange(n_proc):
        proc_samples.append(samples[i * n_proc_smp: i * n_proc_smp + n_proc_smp])

    processes = []

    lock = mp.Lock()
    for i in range(n_proc):
        processes.append(mp.Process(target=generate_fun, args=(proc_samples[i], data_file, lock,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
