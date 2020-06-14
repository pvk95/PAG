from validate import Analyze
from cv_analyze import CV_Analyze
import argparse
import json
import sys
import matplotlib
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
from cv_analyze import fix_object_array

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128


def parse_args():
    parser = argparse.ArgumentParser("Data analysis")
    parser.add_argument('--dirs', nargs='+', required=True,
                        default='experiments/unimodal/', help='All the directories you want')
    parser.add_argument('--config', default='validate/valid_config.json', help='json file')
    parser.add_argument('--save_dir', default='plots/', help='Where to save?')
    parser.add_argument('--gen_cv_metrics', action='store_true')
    parser.add_argument('--metrics', default='metrics_v1.pkl')
    parser.add_argument('--clusters', default='clusters_v1.pkl')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            valid_config = json.load(f)

        assert 'dirs' not in valid_config.keys()
        assert 'save_dir' not in valid_config.keys()
        valid_config['dirs'] = args.dirs
        valid_config['save_dir'] = args.save_dir
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


class Dir_Analyze(object):
    _dir_names = {'ct': 'unimodal', 'ct_attn': 'unimodal+attn', 'PAG-ct': 'PAG:ct',
                  'PAG-ct-pet': 'PAG:ct+pet', 'complete': 'bimodal',
                  'bimodal_attn': 'bimodal+attn'}

    def __init__(self, valid_config):

        self.directories = valid_config['dirs']
        self.save_dir = valid_config['save_dir']
        self.gen_cv_metrics = valid_config['gen_cv_metrics']
        self.metrics = valid_config['metrics']
        self.clusters = valid_config['clusters']

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.root = valid_config['root']
        self.data_path = valid_config['data_path']
        self.tags_consider = valid_config['tags_consider']

        self.cv = CV_Analyze(valid_config)

    def get_metrics(self, n_folds=4):

        directories = self.directories

        parent_conn, child_conn = zip(*[mp.Pipe(duplex=False) for _ in directories])

        assert type(directories) == list

        process = []

        for i, directory in enumerate(directories):
            process.append(mp.Process(target=self.cv.cv_main, args=(directory, child_conn[i], n_folds,)))

        for p in process:
            p.start()

        dir_cv_metrics, dir_ts_metrics, froc_points = list(zip(*[conn.recv() for conn in parent_conn]))

        for p in process:
            p.join()

        dir_cv_metrics = np.array(dir_cv_metrics)
        dir_ts_metrics = np.array(dir_ts_metrics)

        froc_points = None

        return dir_cv_metrics, dir_ts_metrics, froc_points

    def baselines_plot(self, dir_cv_metrics, dir_ts_metrics):

        directories = self.directories

        dir_original_names = [d.split('/')[-2] for d in directories]
        dir_names = [Dir_Analyze._dir_names[d] for d in dir_original_names]

        # cmap = plt.cm.get_cmap('jet')
        # norm = mpl.colors.Normalize(vmin=0, vmax=len(dir_pos))

        detection_cv_metrics = fix_object_array(dir_cv_metrics[:, :, :4, ...])
        overlap_cv_metrics = fix_object_array(dir_cv_metrics[:, :, 4:, ...])

        detection_ts_metrics = fix_object_array(dir_ts_metrics[:, :, :4, ...])
        overlap_ts_metrics = fix_object_array(dir_ts_metrics[:, :, 4:, ...])

        # self.detection_plot(detection_cv_metrics=detection_cv_metrics, detection_ts_metrics=detection_ts_metrics,
        #                     dir_names=dir_names)

        self.dice_plot(overlap_cv_metrics=overlap_cv_metrics[..., :3], overlap_ts_metrics=overlap_ts_metrics[..., :3],
                       dir_names=dir_names)

        self.tag_overlap_plots(overlap_cv_metrics=overlap_cv_metrics[..., 3:],
                               overlap_ts_metrics=overlap_ts_metrics[..., 3:], dir_names=dir_names)

    # =================================================================================
    # ======================== Average Metrics  plot ==================================
    # =================================================================================

    def detection_plot(self, detection_cv_metrics, detection_ts_metrics, dir_names):

        dir_pos = np.arange(len(dir_names))
        metric_names = ['Average sensitivity', 'AUPRC']
        print("Plotting Detection curves (limit)")
        for pos_metric, metric_name in enumerate(metric_names):
            fig, axes = plt.subplots(2, 2, dpi=300, figsize=(18, 12))
            axes = axes.ravel()

            for pos_tag in np.arange(len(axes)):
                ax = axes[pos_tag]
                cv_metric_data = detection_cv_metrics[:, :, pos_tag, pos_metric]
                cv_mean_metric = np.mean(cv_metric_data, axis=1)
                cv_std_metric = np.std(cv_metric_data, axis=1)
                ts_metric_data = detection_ts_metrics[:, :, pos_tag, pos_metric]
                ts_mean_metric = np.mean(ts_metric_data, axis=1)

                for i, pos in enumerate(dir_pos):
                    ax.bar(x=pos, height=cv_mean_metric[i], width=-0.25,
                           yerr=cv_std_metric[i], capsize=5, alpha=1.0, align='edge', label='CV')
                    ax.bar(x=pos, height=ts_mean_metric[i], width=0.25, align='edge', label='Test')
                ax.legend()

                print(ts_mean_metric)

                tag = self.tags_consider[pos_tag]
                if tag == 'mets-tag':
                    y_lim = (0.0, 0.8)
                else:
                    y_lim = (0.4, 1)
                ax.grid(axis='y')
                n_grid_points = np.ceil((y_lim[1] - y_lim[0]) / 0.05 + 1).astype(np.int)
                ax.set_ylim(y_lim[0], y_lim[1])
                ax.set_yticks(np.linspace(y_lim[0], y_lim[1], n_grid_points))
                ax.set_ylabel(metric_name)
                ax.set_xticks(dir_pos)
                ax.set_xticklabels(dir_names)
                ax.set_title('{}: {}'.format(metric_name, self.tags_consider[pos_tag]), fontsize=15)

            plt.savefig(os.path.join(self.save_dir, f'{metric_name}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    def dice_plot(self, overlap_cv_metrics, overlap_ts_metrics, dir_names):

        # =================================================================================
        # ======================== Dice coefficient plot ==================================
        # =================================================================================

        dir_pos = np.arange(len(dir_names))

        print("Plotting overlap curves (limit)")
        plt.rcParams.update({'font.size': 15})
        fig, axes = plt.subplots(3, 1, dpi=300, figsize=(14, 20))

        axes = axes.ravel()

        # limits = [(0.2, 0.8), (0.3, 0.8), (0.4, 0.9)]
        # limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        limits = [(0.4, 0.8), (0.5, 0.9), (0.3, 0.8)]
        ylabels = ['Dice coefficient', 'Precision', 'Recall']
        for ax_num in np.arange(3):

            ax = axes[ax_num]
            cv_metric_data = overlap_cv_metrics[:, :, 0, ax_num]
            cv_metric_mean = np.mean(cv_metric_data, axis=1)
            cv_metric_std = np.std(cv_metric_data, axis=1)

            ts_metric_data = overlap_ts_metrics[:, :, 0, ax_num]
            ts_metric_mean = np.mean(ts_metric_data, axis=1)

            for i, pos in enumerate(dir_pos):
                if i == len(dir_pos) - 1:
                    label = ('CV', 'Test')
                else:
                    label = (None, None)
                ax.bar(x=pos, height=cv_metric_mean[i], width=-0.25,
                       yerr=cv_metric_std[i], capsize=5, align='edge', color='C0', label=label[0])
                ax.bar(x=pos, height=ts_metric_mean[i], width=0.25, align='edge', color='C1',
                       label=label[1])
            # ax.scatter(dir_pos, ts_metric_mean, marker='*', color='r')
            ax.legend(loc=2)

            ax.grid(axis='y')
            ax.set_ylim(limits[ax_num][0], limits[ax_num][1])
            n_ticks = np.int((limits[ax_num][1] - limits[ax_num][0]) / 0.05) + 1
            ax.set_yticks(np.linspace(limits[ax_num][0], limits[ax_num][1], n_ticks))
            ax.set_ylabel(ylabels[ax_num])
            ax.set_xticks(dir_pos)
            ax.set_xticklabels(dir_names)
            ax.tick_params(axis='x', labelsize=18)
            # ax.set_title('Comparison of dice coefficient across models')

        plt.savefig(os.path.join(self.save_dir, 'overlap.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def tag_overlap_plots(self, overlap_cv_metrics, overlap_ts_metrics, dir_names):

        dir_pos = np.arange(len(dir_names))

        print("Plotting tag overlap curves (limit)")
        plt.rcParams.update({'font.size': 15})
        fig, axes = plt.subplots(3, 1, dpi=300, figsize=(14, 20))

        axes = axes.ravel()

        # limits = [(0.6, 1.0), (0.35, 0.95), (0.0, 0.8)]
        # limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        limits = [(0.4, 0.9), (0.1, 0.8), (0, 0.6)]
        ylabels = ['Recall: Primary tumor', 'Recall: Nodular tumor', 'Recall: Metastasis tumor']
        for ax_num in np.arange(3):

            ax = axes[ax_num]
            cv_metric_data = overlap_cv_metrics[:, :, 0, ax_num]
            cv_metric_mean = np.mean(cv_metric_data, axis=1)
            cv_metric_std = np.std(cv_metric_data, axis=1)

            ts_metric_data = overlap_ts_metrics[:, :, 0, ax_num]
            ts_metric_mean = np.mean(ts_metric_data, axis=1)

            for i, pos in enumerate(dir_pos):
                if i == len(dir_pos) - 1:
                    label = ('CV', 'Test')
                else:
                    label = (None, None)
                ax.bar(x=pos, height=cv_metric_mean[i], width=-0.25,
                       yerr=cv_metric_std[i], capsize=5, align='edge', color='C0', label=label[0])
                ax.bar(x=pos, height=ts_metric_mean[i], width=0.25, align='edge', color='C1',
                       label=label[1])
            # ax.scatter(dir_pos, ts_metric_mean, marker='*', color='r')
            ax.legend(loc=2)

            ax.grid(axis='y')
            ax.set_ylim(limits[ax_num][0], limits[ax_num][1])
            n_ticks = np.int(np.around((limits[ax_num][1] - limits[ax_num][0]) / 0.05)) + 1
            ax.set_yticks(np.linspace(limits[ax_num][0], limits[ax_num][1], n_ticks))
            ax.set_ylabel(ylabels[ax_num])
            ax.set_xticks(dir_pos)
            ax.set_xticklabels(dir_names)
            ax.tick_params(axis='x', labelsize=18)
            # ax.set_title('Comparison of dice coefficient across models')

        plt.savefig(os.path.join(self.save_dir, 'overlap-tag.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def dir_main(self):

        dir_cv_metrics, dir_ts_metrics, froc_points = self.get_metrics()

        dir_metrics = {'cv': dir_cv_metrics, 'ts': dir_ts_metrics}
        with open(os.path.join(self.save_dir, 'dir_metrics.pkl'), 'wb') as f:
            pickle.dump(dir_metrics, f)

        self.baselines_plot(dir_cv_metrics=dir_cv_metrics, dir_ts_metrics=dir_ts_metrics)


if __name__ == '__main__':
    valid_config = parse_args()
    dir_analyze = Dir_Analyze(valid_config=valid_config)
    dir_analyze.dir_main()

'''
      axes = axes.reshape((len(self.tags_consider), 2))

      counter = 0

      for pos_tag in np.arange(len(self.tags_consider)):
          for pos_metric in np.arange(2):
              ax = axes[pos_tag, pos_metric]
              y_metric = dir_cv_metrics[:, pos_tag, :, pos_metric]
              # ts_metric = list(zip(*test_metrics[i]))

              ax.bar(x=dir_pos, height=y_metric[:, 0], width=0.1, color=cmap(norm(dir_pos)),
                     yerr=y_metric[:, 1], capsize=5)
              # ax.errorbar(x=dir_pos, y=y_metric[:, 0], yerr=y_metric[:, 1], linewidth=1,
              #            capsize=5, elinewidth=1, markeredgewidth=2, color='black')
              # ax.scatter(dir_pos, ts_metric[0], c='r', marker='+', linewidth=4)
              ax.set_title(self.tags_consider[pos_tag])
              ax.set_xticks([])
              counter += 1

      axes[-1, 0].set_xticks(dir_pos)
      axes[-1, 1].set_xticks(dir_pos)
      axes[-1, 0].set_xticklabels(dir_names)
      axes[-1, 1].set_xticklabels(dir_names)

      '''
'''

fig, ax = plt.subplots(1, 1, figsize=(18, 10), dpi=300)

ax.set_title('Sensitivity vs FP/scan')
ax.set_xlabel('False Positives per scan')
ax.set_ylabel('Sensitivity')
ax.grid()

for i, points in enumerate(froc_points):
    ax.plot(points[0][0], points[1][0], label=dir_names[i])

ax.legend()
plt.savefig('./results_froc.png', dpi=300)

'''
