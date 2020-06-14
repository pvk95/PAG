import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os
import multiprocessing

save_dir = './animations/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def alpha_cmap(cmap, tau=0, alpha=1):
    """
    Create a custom colormap that is fully transparent for
    the `tau`-quantile (inclusive) of the colormap and has
    an alpha channel of `alpha` everywhere else.
    """
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # get the quanitle index
    ix = np.floor(tau * cmap.N).astype(int)
    # Set alpha
    alphas = np.ones(cmap.N) * alpha
    alphas[:ix + 1] = 0  # full transparency for the given quantile
    my_cmap[:, -1] = alphas
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


# plot three animations (CT, PET, Segmentation)
def animate(i, ax1, ax2, ax2_2, ax3, ax3_3, ct, pet, lab):
    """
    NOTE: ax1, ax2, ax3 are global objects.
    """
    # CT
    ax1.set_data(ct[i])
    # CT + PET
    ax2.set_data(ct[i])
    ax2_2.set_data(pet[i])
    # CT + Segmentation
    ax3.set_data(ct[i])
    ax3_3.set_data(lab[i])
    return ax1, ax2, ax3,


def generateAnimation(days, id):
    index = 1
    for curr_day in days:
        sub_folders = glob.glob(PATH_DATADIR + f'{curr_day}/*/')
        for curr_folder in sub_folders:
            data = np.load(curr_folder + 'lsa.npz')
            ct = data['CT']
            pet = data['PET']
            lab = data['Labels']

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=40)
            axes[0].set_title("CT")
            ax1 = axes[0].imshow(ct[0], vmin=-1000, vmax=1000, cmap=plt.cm.gray)
            axes[1].set_title("CT + PET")
            ax2 = axes[1].imshow(ct[0], vmin=-1000, vmax=1000, cmap=plt.cm.gray)
            ax2_2 = axes[1].imshow(pet[0], vmin=0, vmax=7, cmap=plt.cm.gist_heat, alpha=0.5)
            axes[2].set_title("CT + Segmentation")
            ax3 = axes[2].imshow(ct[0], vmin=-1000, vmax=1000, cmap=plt.cm.gray)
            ax3_3 = axes[2].imshow(lab[0], vmin=lab.min(), vmax=lab.max(),
                                   cmap=alpha_cmap(plt.cm.hsv, 0, 0.7))
            # create and save the animation
            ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, ct.shape[0]),
                                          interval=100, repeat_delay=1000,
                                          fargs=(ax1, ax2, ax2_2, ax3, ax3_3, ct, pet, lab))
            ani.save(save_dir + f'data_{id}_{index}.mp4', 'FFMpegFileWriter')

            plt.close()

            index = index + 1

            del ct
            del pet
            del lab


PATH_DATADIR = 'data/'
days = [f'day{i}' for i in range(2, 31)]

n_threads = 15
n_days = len(days)
lim = np.ceil(n_days / n_threads).astype(np.int)
days_proc = []
counter = 0
i = 0

while counter < n_threads:
    days_proc.append(days[i:i + lim])
    i = i + lim
    counter += 1

prs = []
for j in range(len(days_proc)):
    temp = multiprocessing.Process(target=generateAnimation, args=(days_proc[j], j + 1))
    prs.append(temp)
    print(f"Process {j} started")
    prs[-1].start()

for p in prs:
    p.join()

# get_ipython().system('zip -r animations.zip animations/')
