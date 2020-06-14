from validate import Analyze

import matplotlib
import os
import multiprocessing as mp
import argparse
import sys
import json
import numpy as np
import pickle

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128


def fix_object_array(np_arr):
    assert np_arr.dtype == np.object

    shape = np_arr.shape
    if not isinstance(np_arr.ravel()[0], tuple):
        return np.array(np_arr, dtype=np.float)
    else:
        n_points = (len(np_arr.ravel()[0]),)

        return np.reshape(np.array([np.array(p) for p in np_arr.ravel()], dtype=np.float),
                          newshape=(shape + n_points))


def parse_args():
    parser = argparse.ArgumentParser("CV")
    parser.add_argument('--dir', required=True,
                        default='experiments/unimodal/', help='All the directories you want')
    parser.add_argument('--n_folds', default=4, type=int, help='No. of cv folds')
    parser.add_argument('--config', default='validate/valid_config.json', help='json file')
    parser.add_argument('--gen_cv_metrics', action='store_true')
    parser.add_argument('--metrics', default='metrics_v1.pkl')
    parser.add_argument('--clusters', default='clusters_v1.pkl')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            valid_config = json.load(f)

        valid_config['dir'] = args.dir
        valid_config['n_folds'] = args.n_folds
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


class CV_Analyze(Analyze):

    def __init__(self, valid_config):

        super(Analyze, self).__init__()

        self.root = valid_config['root']
        self.data_path = valid_config['data_path']
        self.tags_consider = valid_config['tags_consider']

        self.detect = valid_config['detect']
        self.train = valid_config['train']
        self.valid = valid_config['valid']
        self.test = valid_config['test']
        self.plot_curves = valid_config['plot_curves']

        self.gen_cv_metrics = valid_config['gen_cv_metrics']
        self.metrics = valid_config['metrics']
        self.clusters = valid_config['clusters']

    def log(self, metrics):

        detection_metrics = fix_object_array(metrics[:, :4])
        overlap_metrics = fix_object_array(metrics[:, 4:])

        print(overlap_metrics.shape)

        # cases = ['detect', 'overlap']
        cases = ['overlap']

        for c in cases:
            if c == 'detect':
                metric_names = ["Average Sensitivity", "AUPRC"]
                consider_metrics = detection_metrics
                tags = self.tags_consider[:4]
            else:
                metric_names = ['Dice coefficient', 'Precision', 'Recall',
                                'Recall_tumor', 'Recall_node', 'Recall_mets']
                consider_metrics = overlap_metrics
                tags = self.tags_consider[4:]

            metrics_mean = np.mean(consider_metrics, axis=0)
            metrics_std = np.std(consider_metrics, axis=0)

            for i in range(len(metrics_mean)):
                t = tags[i]
                print(f"==== {t} ====")
                for j, m in enumerate(metric_names):
                    print("\t{} --> {}".format(m, consider_metrics[:, i, j]))
                    print("\t{} --> Mean: {}, Std: {}".format(m, metrics_mean[i, j], metrics_std[i, j]))

    def get_cv_metrics(self, directories, parent_conn, child_conn, verbose=False):

        processes = []

        print(directories)
        for i, present_directory in enumerate(directories):
            processes.append(mp.Process(target=self.analyze_fun, args=(present_directory, child_conn[i],)))

        for p in processes:
            p.start()

        metrics, froc_points = list(zip(*[conn.recv() for conn in parent_conn]))

        for p in processes:
            p.join()

        metrics = np.array(metrics)

        if verbose:
            self.log(metrics)

        return metrics, froc_points

    def cv_main(self, directory, dir_conn=None, n_folds=4, verbose=False):

        folders = [f'cv{i}/' for i in range(n_folds)]
        valid_dirs = []
        for f in folders:
            f_path = os.path.join(self.root, directory, f)
            # print(f_path)
            if os.path.exists(f_path):
                valid_dirs.append(f_path)

        if self.gen_cv_metrics:
            print("################ Metrics by each fold (Validation data) ################")
            self.valid = True
            parent_connections, child_connections = zip(*[mp.Pipe(duplex=False) for _ in valid_dirs])
            cv_metrics, _ = self.get_cv_metrics(directories=valid_dirs, parent_conn=parent_connections,
                                                child_conn=child_connections, verbose=verbose)
            self.valid = False
            metrics = {'cv_metrics': cv_metrics}

            print("################ Metrics by complete training data (Test data) ################")
            test_dir = [os.path.join(self.root, directory, 'ts/')]
            self.test = True
            parent_connections, child_connections = zip(*[mp.Pipe(duplex=False) for _ in test_dir])
            ts_metrics, froc_points = self.get_cv_metrics(directories=test_dir,
                                                          parent_conn=parent_connections,
                                                          child_conn=child_connections, verbose=verbose)
            self.test = False
            metrics['ts_metrics'] = ts_metrics
            metrics['froc_points'] = froc_points

            print("################ Metrics by each fold (Test data) ################")
            self.test = True
            parent_connections, child_connections = zip(*[mp.Pipe(duplex=False) for _ in valid_dirs])
            ts_cv_metrics, _ = self.get_cv_metrics(directories=valid_dirs, parent_conn=parent_connections,
                                                   child_conn=child_connections, verbose=verbose)
            self.test = False
            metrics['ts_cv_metrics'] = ts_cv_metrics

            with open(os.path.join(self.root, directory, self.metrics), 'wb') as f:
                pickle.dump(metrics, f)

        else:
            with open(os.path.join(self.root, directory, self.metrics), 'rb') as f:
                metrics = pickle.load(f)

                print("Loaded metrics for directory: ", directory)
                cv_metrics = metrics['cv_metrics']
                ts_metrics = metrics['ts_metrics']
                ts_cv_metrics = metrics['ts_cv_metrics']
                froc_points = None

                if verbose:
                    print("+++++++++++ Metrics by each fold (Validation data) +++++++++++")
                    self.log(cv_metrics)

                    print("+++++++++++ Metrics by complete training data (Test data) +++++++++++")
                    self.log(ts_metrics)

                    print("+++++++++++ Metrics by each fold (Test data) +++++++++++")
                    self.log(ts_cv_metrics)

        if dir_conn is not None:
            dir_conn.send((cv_metrics, ts_metrics, froc_points))


if __name__ == '__main__':
    valid_config = parse_args()
    cv = CV_Analyze(valid_config=valid_config)
    dir = valid_config['dir']
    n_folds = valid_config['n_folds']
    cv.cv_main(verbose=True, directory=dir, n_folds=n_folds)
