import argparse
import json
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import copy

from dir_analyze import Dir_Analyze
from cv_analyze import fix_object_array

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128


def parse_args():
    parser = argparse.ArgumentParser("Data analysis")
    parser.add_argument('--config', default='validate/valid_config.json', help='json file')
    parser.add_argument('--n_fracs', type=int, default=6, help='No. of fractions')
    parser.add_argument('--save_dir', default='plots/', help='Where to save?')
    parser.add_argument('--gen_cv_metrics', action='store_true')
    parser.add_argument('--metrics', default='metrics_v1.pkl')
    parser.add_argument('--clusters', default='clusters_v1.pkl')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            valid_config = json.load(f)

        assert 'save_dir' not in valid_config.keys()
        valid_config['save_dir'] = args.save_dir
        valid_config['n_fracs'] = args.n_fracs
        valid_config['gen_cv_metrics'] = args.gen_cv_metrics
        valid_config['metrics'] = args.metrics
        valid_config['clusters'] = args.clusters

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config))
        sys.exit(1)
    except json.JSONDecodeError:
        print("ERROR: Config file is not a valid JSON file!")
        sys.exit(1)

    return valid_config


class FRA(object):
    _fractions = {'frac_0': 0.03, 'frac_1': 0.05, 'frac_2': 0.1, 'frac_3': 0.15,
                  'frac_4': 0.3, 'frac_5': 0.5, 'complete': 1.0}

    def __init__(self, valid_config):

        self.tags_consider = valid_config['tags_consider']
        self.n_fracs = valid_config['n_fracs']
        self.save_dir = valid_config['save_dir']
        self.gen_cv_metrics = valid_config['gen_cv_metrics']
        self.metrics = valid_config['metrics']
        self.clusters = valid_config['clusters']

        self.valid_config = valid_config

        self.folders = None
        self.dir_pos = None
        self.colors = None

        self.init_params()

    def init_params(self):

        self.folders = [f'frac_{i}' for i in range(self.n_fracs)] + ['complete']

        self.dir_pos = [FRA._fractions[f] for f in self.folders]

        self.colors = {'PAG:ct': 'green', 'PAG:ct+pet': 'blue', 'Bimodal': 'red',
                       'Unimodal:limit': 'green', 'Bimodal:limit': 'red'}

    def generate_metrics(self):

        root_unimodal = 'multi-baselines/'  # Baseline unimodal.
        # root_bilim = 'multi-baselines/'  # Limit of Bimodal

        root_pag = 'pag/less'
        root_bimodal = 'pag/fractions'

        # Unimodal
        unimodal_folder = [os.path.join(root_unimodal, 'ct')]

        # Frac_Bimodal
        bimodal_folders = [os.path.join(root_bimodal, f) for f in self.folders]

        # Bimodal at Limit (pet/ct=1)
        # bilim_folder = [os.path.join(root_bilim, 'bimodal')]
        bilim_folder = [bimodal_folders[-1]]

        # PAG
        pet_pag_folders = [os.path.join(root_pag, f, 'PAG-ct-pet') for f in self.folders]
        ct_pag_folders = [os.path.join(root_pag, f, 'PAG-ct') for f in self.folders]

        models_consider = {'Unimodal:limit': unimodal_folder, 'Bimodal:limit': bilim_folder, 'PAG:ct': ct_pag_folders,
                           'PAG:ct+pet': pet_pag_folders, 'Bimodal': bimodal_folders, }

        models_cv_metrics = {'Unimodal:limit': None, 'Bimodal:limit': None, 'PAG:ct': None,
                             'PAG:ct+pet': None, 'Bimodal': None}
        models_ts_metrics = {'Unimodal:limit': None, 'Bimodal:limit': None, 'PAG:ct': None,
                             'PAG:ct+pet': None, 'Bimodal': None}

        '''

        models_consider = {'Unimodal-limit': unimodal_folder, 'Bimodal-limit': bilim_folder,
                           'Bimodal': bimodal_folders}

        models_cv_metrics = {'Unimodal-limit': None, 'Bimodal-limit': None, 'Bimodal': None}
        models_ts_metrics = {'Unimodal-limit': None, 'Bimodal-limit': None, 'Bimodal': None}
        
        '''
        for k in models_consider.keys():
            valid_config = copy.deepcopy(self.valid_config)
            assert 'dirs' not in valid_config.keys()
            valid_config['dirs'] = models_consider[k]

            n_folds = 4
            dir_analyze = Dir_Analyze(valid_config=valid_config)

            dir_cv_metrics, dir_ts_metrics, _ = dir_analyze.get_metrics(n_folds=n_folds)

            models_cv_metrics[k] = dir_cv_metrics
            models_ts_metrics[k] = dir_ts_metrics

        models_metrics = {'cv': models_cv_metrics, 'ts': models_ts_metrics}

        print(models_metrics.keys())
        print(models_metrics['cv'].keys())
        print(models_metrics['ts'].keys())

        with open(os.path.join(self.save_dir, 'models_metrics.pkl'), 'wb') as f:
            pickle.dump(models_metrics, f)

    def plot_detection_metrics(self, models_cv_metrics, models_ts_metrics):

        x_lim = 1.1
        spacing = 0.1
        x_ticks = np.arange(0, x_lim + 0.001, spacing)

        metric_names = ['Average Sensitivity', 'AUPRC']

        print("Plotting Metrics curve ...")
        for pos_metric, metric_name in enumerate(metric_names):

            fig, axes = plt.subplots(2, 2, dpi=300, figsize=(20, 10))
            plt.rcParams.update({'font.size': 12})

            axes = axes.ravel()
            for pos_tag in np.arange(len(axes)):
                ax = axes[pos_tag]

                for i, k in enumerate(models_cv_metrics.keys()):
                    dir_cv_metrics = models_cv_metrics[k]
                    dir_ts_metrics = models_ts_metrics[k]
                    cv_detection_metrics = fix_object_array(np_arr=dir_cv_metrics[:, :, :4, ...])
                    ts_detection_metrics = fix_object_array(np_arr=dir_ts_metrics[:, :, :4, ...])

                    cv_dice_metric = cv_detection_metrics[:, :, pos_tag, pos_metric]
                    y_metric = np.stack([np.mean(cv_dice_metric, axis=1), np.std(cv_dice_metric, axis=1)], axis=-1)
                    ts_dice_metric = ts_detection_metrics[:, :, pos_tag, pos_metric]

                    if k == 'Bimodal-limit':
                        assert y_metric.shape[0] == 1
                        y_metric = np.repeat(y_metric, repeats=len(x_ticks), axis=0)

                        min_val = y_metric[:, 0] - y_metric[:, 1]
                        max_val = y_metric[:, 0] + y_metric[:, 1]
                        ax.plot(x_ticks, y_metric[:, 0], linestyle='dotted', color=self.colors[k], label=k)
                        ax.fill_between(x=x_ticks, y1=min_val, y2=max_val, facecolor=self.colors[k], alpha=0.2)
                    elif k == 'PAG-ct' or k == 'Unimodal-limit':
                        continue
                    else:
                        ax.errorbar(x=self.dir_pos, y=y_metric[:, 0], yerr=y_metric[:, 1], linewidth=1,
                                    capsize=4, elinewidth=1, markeredgewidth=2, color=self.colors[k])
                        ax.scatter(self.dir_pos, y_metric[:, 0], c=self.colors[k], marker='o', linewidth=1, label=k)
                        ax.scatter(self.dir_pos, np.mean(ts_dice_metric, axis=1), c=self.colors[k], marker='*')

                ax.set_xlim(0, x_lim)
                ax.set_xticks(x_ticks)
                ax.set_title('{}: {}'.format(metric_name, self.tags_consider[pos_tag]), fontsize=15)
                ax.set_xlabel('Fraction of total PET images', fontsize=18)
                ax.set_ylabel(metric_name)
                ax.grid()
                ax.legend()

            plt.savefig(os.path.join(self.save_dir, f'fra_{metric_name}_temp.png'), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_dice_metrics(self, models_cv_metrics, models_ts_metrics):

        x_lim = 1.1
        spacing = 0.05
        x_ticks = np.arange(0, x_lim + 0.001, spacing)

        plt.rcParams.update({'font.size': 15})

        n_metrics = 3
        assert n_metrics == 3
        metric_names = ['Dice coefficient', 'Precision', 'Recall']
        # limits = [(0.35, 0.8), (0.3, 0.8), (0.4, 0.85)]
        # limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        limits = [(0.3, 0.9), (0.4, 1.0), (0.2, 0.8)]
        for metric_num in np.arange(n_metrics):
            fig, axes = plt.subplots(1, 1, dpi=300, figsize=(18, 13))
            ax = axes
            print("Plotting {} curve ...".format(metric_names[metric_num]))

            for i, k in enumerate(models_cv_metrics.keys()):
                dir_cv_metrics = models_cv_metrics[k]
                dir_ts_metrics = models_ts_metrics[k]
                cv_overlap_metrics = fix_object_array(np_arr=dir_cv_metrics[:, :, 4:, ...])
                ts_overlap_metrics = fix_object_array(np_arr=dir_ts_metrics[:, :, 4:, ...])

                cv_metric = cv_overlap_metrics[:, :, 0, metric_num]
                ts_metric = ts_overlap_metrics[:, :, 0, metric_num]
                y_metric = np.stack([np.mean(cv_metric, axis=1), np.std(cv_metric, axis=1)], axis=-1)

                if k == 'Unimodal:limit' or k == 'Bimodal:limit':

                    y_metric = np.repeat(y_metric, repeats=len(x_ticks), axis=0)
                    min_val = y_metric[:, 0] - y_metric[:, 1]
                    max_val = y_metric[:, 0] + y_metric[:, 1]
                    ax.plot(x_ticks, y_metric[:, 0], linestyle='dotted', color=self.colors[k], label=k)
                    ax.fill_between(x=x_ticks, y1=min_val, y2=max_val, facecolor=self.colors[k], alpha=0.2)
                elif k == 'PAG:ct':
                    continue
                else:

                    ax.errorbar(x=self.dir_pos, y=y_metric[:, 0], yerr=y_metric[:, 1], linewidth=1,
                                capsize=5, elinewidth=1, markeredgewidth=2, color=self.colors[k])
                    ax.scatter(self.dir_pos, y_metric[:, 0], c=self.colors[k], marker='o', linewidth=1,
                               label=k + ' (CV)')
                    ax.scatter(self.dir_pos, np.mean(ts_metric, axis=1), c=self.colors[k], marker='*', linewidth=4,
                               label=k + ' (Test)')

            ax.set_xticks(x_ticks)
            ax.set_xlim(0, x_lim)
            ax.set_ylim(limits[metric_num][0], limits[metric_num][1])
            ax.set_yticks(np.arange(limits[metric_num][0], limits[metric_num][1], 0.05))
            # ax.set_title('Dice coefficient vs fraction of total PET images', fontsize=15)
            ax.set_ylabel(metric_names[metric_num])
            ax.set_xlabel('Fraction of total PET images', fontsize=18)

            ax.grid()
            ax.legend(loc=4)

            plt.savefig(os.path.join(self.save_dir, f'fra_{metric_names[metric_num]}.png'), bbox_inches='tight',
                        dpi=300)
            plt.close()

    def plot_tag_overlap(self, models_cv_metrics, models_ts_metrics):

        x_lim = 1.1
        spacing = 0.05
        x_ticks = np.arange(0, x_lim + 0.001, spacing)

        plt.rcParams.update({'font.size': 15})

        n_prev_metrics = 3
        metric_names = ['Recall (Primary tumor)', 'Recall (Nodular tumor)', 'Recall (Metastasis tumor)']
        # limits = [(0.4, 0.95), (0.45, 0.9), (0.0, 0.75)]
        # limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        limits = [(0.2, 0.9), (0.2, 0.8), (0.0, 0.7)]

        for metric_num in np.arange(n_prev_metrics, n_prev_metrics + len(metric_names)):
            fig, axes = plt.subplots(1, 1, dpi=300, figsize=(18, 13))
            ax = axes
            print("Plotting {} curve ...".format(metric_names[metric_num-n_prev_metrics]))

            for i, k in enumerate(models_cv_metrics.keys()):
                dir_cv_metrics = models_cv_metrics[k]
                dir_ts_metrics = models_ts_metrics[k]
                cv_overlap_metrics = fix_object_array(np_arr=dir_cv_metrics[:, :, 4:, ...])
                ts_overlap_metrics = fix_object_array(np_arr=dir_ts_metrics[:, :, 4:, ...])

                cv_metric = cv_overlap_metrics[:, :, 0, metric_num]
                ts_metric = ts_overlap_metrics[:, :, 0, metric_num]
                y_metric = np.stack([np.mean(cv_metric, axis=1), np.std(cv_metric, axis=1)], axis=-1)

                if k == 'Unimodal:limit' or k == 'Bimodal:limit':

                    y_metric = np.repeat(y_metric, repeats=len(x_ticks), axis=0)
                    min_val = y_metric[:, 0] - y_metric[:, 1]
                    max_val = y_metric[:, 0] + y_metric[:, 1]
                    ax.plot(x_ticks, y_metric[:, 0], linestyle='dotted', color=self.colors[k], label=k)
                    ax.fill_between(x=x_ticks, y1=min_val, y2=max_val, facecolor=self.colors[k], alpha=0.2)
                elif k == 'PAG:ct':
                    continue
                else:

                    ax.errorbar(x=self.dir_pos, y=y_metric[:, 0], yerr=y_metric[:, 1], linewidth=1,
                                capsize=5, elinewidth=1, markeredgewidth=2, color=self.colors[k])
                    ax.scatter(self.dir_pos, y_metric[:, 0], c=self.colors[k], marker='o', linewidth=1,
                               label=k + ' (CV)')
                    ax.scatter(self.dir_pos, np.mean(ts_metric, axis=1), c=self.colors[k], marker='*', linewidth=4,
                               label=k + ' (Test)')

            ax.set_xticks(x_ticks)
            ax.set_xlim(0, x_lim)
            ax.set_ylim(limits[metric_num-n_prev_metrics][0], limits[metric_num-n_prev_metrics][1])
            ax.set_yticks(np.arange(limits[metric_num-n_prev_metrics][0], limits[metric_num-n_prev_metrics][1], 0.05))
            # ax.set_title('Dice coefficient vs fraction of total PET images', fontsize=15)
            ax.set_ylabel(metric_names[metric_num-n_prev_metrics])
            ax.set_xlabel('Fraction of total PET images', fontsize=18)

            ax.grid()
            ax.legend(loc=4)

            plt.savefig(os.path.join(self.save_dir, f'fra_{metric_names[metric_num-n_prev_metrics]}.png'),
                        bbox_inches='tight',
                        dpi=300)
            plt.close()

    def plot_dice_bbox(self, models_cv_metrics):

        x_lim = 1.1
        spacing = 0.05
        x_ticks = np.arange(0, x_lim + 0.001, spacing)

        fig, axes = plt.subplots(2, 1, dpi=300, figsize=(18, 12), gridspec_kw={'height_ratios': [0.5, 1]})
        axes = axes.ravel()
        plt.rcParams.update({'font.size': 15})

        print(self.dir_pos)

        print("Plotting Dice box curve ...")
        for i, k in enumerate(models_cv_metrics.keys()):
            dir_cv_metrics = models_cv_metrics[k]
            overlap_metrics = fix_object_array(np_arr=dir_cv_metrics[:, :, 4:, ...])

            dice_metric = overlap_metrics[:, :, 0, 0]

            if k == 'Bimodal-limit':
                for ax in axes:
                    y_metric = np.stack([np.mean(dice_metric, axis=1), np.std(dice_metric, axis=1)], axis=-1)

                    y_metric = np.repeat(y_metric, repeats=len(x_ticks), axis=0)
                    min_val = y_metric[:, 0] - y_metric[:, 1]
                    max_val = y_metric[:, 0] + y_metric[:, 1]
                    ax.plot(x_ticks, y_metric[:, 0], linestyle='dotted', color=self.colors[k], label=k)
                    ax.fill_between(x=x_ticks, y1=min_val, y2=max_val, facecolor=self.colors[k], alpha=0.2)

            elif k == 'PAG-ct' or k == 'Unimodal-limit':
                continue
            elif k == 'PAG-ct-pet' or k == 'Bimodal':

                ax = axes[0] if k == 'PAG-ct-pet' else axes[1]
                y_lim = (0.5, 0.75) if k == 'PAG-ct-pet' else (0.3, 0.75)
                ax.set_ylim(y_lim[0], y_lim[1])
                ax.set_yticks(np.arange(y_lim[0], y_lim[1], 0.05))

                data_plot = [dice_metric[pos, :] for pos in range(7)]
                bp = ax.boxplot(data_plot, patch_artist=True, vert=True, widths=0.01, positions=self.dir_pos,
                                manage_ticks=False)
                ax.plot(self.dir_pos, np.mean(dice_metric, axis=1), c=self.colors[k], label=k)
                for patch in bp['boxes']:
                    patch.set(facecolor=self.colors[k])

        for ax in axes:
            ax.set_xticks(x_ticks)
            ax.set_xlim(0, x_lim)

            ax.set_ylabel('Dice coefficient')
            ax.set_xlabel('Fraction of total PET images')

            ax.grid()
            ax.legend(loc=4)

        axes[0].set_title('Dice coefficient vs fraction of total PET images', fontsize=15)
        plt.savefig(os.path.join(self.save_dir, 'fra_dice_box.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def fractions_plot(self):

        # if not os.path.exists(os.path.join(self.save_dir, 'models_metrics.pkl')):
        self.generate_metrics()

        with open(os.path.join(self.save_dir, 'models_metrics.pkl'), 'rb') as f:
            models_metrics = pickle.load(f)

            models_cv_metrics = models_metrics['cv']
            models_ts_metrics = models_metrics['ts']

        # self.plot_detection_metrics(models_cv_metrics=models_cv_metrics, models_ts_metrics=models_ts_metrics)
        self.plot_dice_metrics(models_cv_metrics=models_cv_metrics, models_ts_metrics=models_ts_metrics)
        # self.plot_dice_bbox(models_cv_metrics=models_cv_metrics)
        self.plot_tag_overlap(models_cv_metrics=models_cv_metrics, models_ts_metrics=models_ts_metrics)

    def fra_main(self):
        self.fractions_plot()


if __name__ == '__main__':
    valid_config = parse_args()

    fra = FRA(valid_config=valid_config)
    fra.fra_main()
