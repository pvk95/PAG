from analysis_utils import *

import h5py
import matplotlib
import os
import multiprocessing as mp
import argparse
import pickle
import json

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128


class Analyze(object):

    def __init__(self, valid_config):

        self.root = valid_config['root']
        self.data_path = valid_config['data_path']
        self.tags_consider = valid_config['tags_consider']

        self.detect = valid_config['detect']
        self.train = valid_config['train']
        self.valid = valid_config['valid']
        self.test = valid_config['test']
        self.plot_curves = valid_config['plot_curves']
        self.clusters = valid_config['clusters']

    def animate(self, save_dir):
        def generate_anim(data_type, data_indexes):

            data_file_name = 'seg_predictions.h5' if data_type == 'test' else 'predictions.h5'

            if not os.path.exists(save_dir + f'{data_type}/'):
                os.makedirs(os.path.join(save_dir, data_type))

            with h5py.File(data_path + 'data_file.h5', 'r') as file:
                data = file['data/data'][data_indexes, ...]

            n_samples = data.shape[0]

            for i in range(n_samples):
                data_idx = data_indexes[i]
                print("Rendering image: {} ...".format(data_idx))
                mov_name = save_dir + '{}/mov_{}.mp4' \
                                      ''.format(data_type, data_idx)

                img = data[i, ...]

                preds = [None] * 3
                network_names = ['ct', 'pet', 'mask']

                with h5py.File(save_dir + data_file_name, 'r') as file:
                    for j in range(len(preds)):
                        try:
                            name = '{}/{}'.format(network_names[j], data_type)
                            preds[j] = file[name][()]
                        except KeyError:
                            pass

                img_recon = np.zeros_like(img) if preds[0] is None else preds[0][i, :, :, :, :]
                pet_recon = np.zeros_like(img) if preds[1] is None else preds[1][i, :, :, :, :]
                seg_predict = np.zeros_like(img) if preds[2] is None else preds[2][i, :, :, :, :]
                render(clusters, img, img_recon, pet_recon, seg_predict, data_idx, mov_name)

        train_idxs, valid_idxs, test_idxs = train_valid_test_idxs(save_dir)

        try:
            with open(save_dir + self.clusters, 'rb') as my_dict:
                clusters = pickle.load(my_dict)
        except FileNotFoundError:
            print("Ensure that the clusters pickle file is generated ...")
            sys.exit(1)

        data_path = self.data_path
        train = self.train
        valid = self.valid
        test = self.test

        if train:
            generate_anim(data_type='train', data_indexes=train_idxs)
        if valid:
            generate_anim(data_type='valid', data_indexes=valid_idxs)
        if test:
            generate_anim(data_type='test', data_indexes=test_idxs)

    def detect_fn(self, save_dir):
        def sub_detect(data_type, data_indexes, extra_indices=None):

            if len(data_indexes) == 0:
                return

            with h5py.File(self.data_path + 'data_file.h5', 'r') as file:
                data = file['data/data'][data_indexes, ...]

            data_file_name = 'seg_predictions.h5' if data_type == 'test' else 'predictions.h5'
            with h5py.File(os.path.join(save_dir, data_file_name), 'r') as file:
                print("For {} loading from {}".format(data_type, data_file_name))
                predictions = file[f'mask/{data_type}'][()]

            if extra_indices is not None:
                try:
                    with h5py.File(os.path.join(save_dir, 'seg_predictions.h5'), 'r') as file:
                        extra_predictions = file[f'extra/mask/{data_type}'][()]
                    predictions = np.concatenate((predictions, extra_predictions))

                    with h5py.File(self.data_path + 'data_file.h5', 'r') as file:
                        data = np.concatenate((data, file['data/extra_data'][()]))

                    data_indexes = np.concatenate((data_indexes, extra_indices))
                except KeyError:
                    print("Extra predictions not found. Skipping ...")

            dice, precision, recall = dice_score(seg_predict=predictions, seg_mask=data[..., -1])

            recall_tumor = reconstruct_mask(seg_predict=predictions, tag_consider='tumor-tag', meta_data=meta_data,
                                            index_data=index_data, data_indexes=data_indexes, data=data)
            recall_node = reconstruct_mask(seg_predict=predictions, tag_consider='node-tag', meta_data=meta_data,
                                           index_data=index_data, data_indexes=data_indexes, data=data)
            recall_mets = reconstruct_mask(seg_predict=predictions, tag_consider='mets-tag', meta_data=meta_data,
                                           index_data=index_data, data_indexes=data_indexes, data=data)
            '''
            gt_cluster = clusters['gt_cluster']
            model_cluster = clusters['model_cluster']

            n_samples = data.shape[0]

            for i in range(n_samples):
                data_idx = data_indexes[i]

                print("Processing image {}".format(data_idx))
                bboxes = get_gt_bbox(data_idx=data_idx, meta_data=meta_data, index_data=index_data)
                gt_cluster[data_idx] = bboxes

                model_bboxes, _, _, = merge_labels(image=(predictions[i, 0, :, :, :]), pred=True)

                model_cluster[data_idx] = model_bboxes

            gt_cluster, model_cluster = get_hits(gt_cluster=gt_cluster,
                                                 model_cluster=model_cluster, data_indexes=data_indexes)

            clusters['gt_cluster'] = gt_cluster
            clusters['model_cluster'] = model_cluster
            '''

            clusters['{}_overlap'.format(data_type)] = (dice, precision, recall, recall_tumor, recall_node, recall_mets)

        if not self.detect:
            return None

        print("Generating clusters.pkl")

        try:
            with open(os.path.join(save_dir, self.clusters), 'rb') as my_dict:
                clusters = pickle.load(my_dict)
                print("Loaded clusters")
        except FileNotFoundError:
            clusters = {'gt_cluster': {}, 'model_cluster': {}, 'valid_dice': {},
                        'valid_overlap': {}, 'test_overlap': {}}

        with open(os.path.join(self.data_path, 'meta_data.pkl'), 'rb') as my_dict:
            meta_data = pickle.load(my_dict)

        with open(os.path.join(self.data_path, 'index_data.pkl'), 'rb') as my_dict:
            index_data = pickle.load(my_dict)

        train_idxs, valid_idxs, test_idxs = train_valid_test_idxs(save_dir=save_dir)

        if self.train:
            sub_detect(data_type='train', data_indexes=train_idxs)
        if self.valid:
            sub_detect(data_type='valid', data_indexes=valid_idxs)
        if self.test:
            sub_detect(data_type='test', data_indexes=test_idxs, extra_indices=np.arange(337, 397))
            # sub_detect(data_type='test', data_indexes=test_idxs)
        with open(os.path.join(save_dir, self.clusters), 'wb') as file:
            pickle.dump(clusters, file)

    def calculate_metrics(self, save_dir, conn=None):
        def sub_calc_metrics(tag='all_dice'):

            if not os.path.exists(os.path.join(save_dir, data_type)):
                os.makedirs(os.path.join(save_dir, data_type))

            if tag == 'overlap':
                try:
                    result = clusters['{}_overlap'.format(data_type)]
                    dice, precision, recall, recall_tumor, recall_node, recall_mets = result

                    with open(os.path.join(save_dir, data_type, f'metrics_{tag}.txt'), 'w') as file:
                        file.write("\n\n====== Overlap metrics: {} ======\n".format(result))

                    print("{}: {}".format(tag, result))
                    return (dice, precision, recall, recall_tumor, recall_node, recall_mets), None

                except KeyError:
                    print('{}_dice not found'.format(data_type))
                    return (0,), None
                except Exception as e:
                    print("Exception {} occcured for {}".format(e, save_dir))
            else:
                '''
                gt_cluster = clusters['gt_cluster']
                model_cluster = clusters['model_cluster']

                score, auprc, precision_points, recall_points, froc_points = get_metrics(gt_cluster=gt_cluster,
                                                                                         model_cluster=model_cluster,
                                                                                         data_indexes=data_indexes,
                                                                                         tag=tag,
                                                                                         froc=True)

                if self.plot_curves:
                    plot_froc(fig_name=os.path.join(save_dir, data_type, f'FROC_{tag}.png'),
                              x_points=froc_points[0], y_points=froc_points[1])

                    plot_prc(precision_points=precision_points, recall_points=recall_points, auprc=auprc,
                             fig_name=os.path.join(save_dir, data_type, f'PRC_{tag}.png'))

                with open(os.path.join(save_dir, data_type, f'metrics_{tag}.txt'), 'w') as file:
                    file.write("\n\n====== Average sensitivity: {} ======\n".format(score))
                    file.write("====== AUPRC: {} ======\n".format(auprc))

                print("{}: {}, {}".format(tag, score, auprc))
                return (np.round(score, 3), np.round(auprc, 3)), froc_points
                '''
                return (np.round(0.5, 3), np.round(0.5, 3)), None

        try:
            with open(os.path.join(save_dir, self.clusters), 'rb') as my_dict:
                clusters = pickle.load(my_dict)
        except FileNotFoundError as e:
            print(e)
            print("======== Detection not performed. Exiting ========")
            sys.exit(1)
        except Exception as e:
            print("{} occurred for {}".format(e, save_dir))

        train_idxs, valid_idxs, test_idxs = train_valid_test_idxs(save_dir=save_dir)

        if self.train:
            data_type = 'train'
            data_indexes = train_idxs
        elif self.valid:
            data_type = 'valid'
            data_indexes = valid_idxs
        elif self.test:
            data_type = 'test'
            data_indexes = test_idxs

        metrics, froc_points = list(zip(*[sub_calc_metrics(tag=t) for t in self.tags_consider]))

        metrics = np.array(metrics, dtype=np.object)

        if conn is not None:
            conn.send((metrics, froc_points))
            conn.close()

    def analyze_fun(self, save_dir, conn=None):

        self.detect_fn(save_dir=save_dir)

        if self.plot_curves:
            print("======== Plotting training curves  ========")
            plot_training_curves(save_dir=save_dir)

        print("======== Calculating metrics ========")
        if self.train + self.test + self.valid <= 1:
            self.calculate_metrics(save_dir=save_dir, conn=conn)


class Validate(Analyze):

    def __init__(self):
        super(Analyze, self).__init__()

        valid_config = self.parse_args()

        self.root = valid_config['root']
        self.data_path = valid_config['data_path']
        self.tags_consider = valid_config['tags_consider']

        self.detect = valid_config['detect']
        self.train = valid_config['train']
        self.valid = valid_config['valid']
        self.test = valid_config['test']
        self.plot_curves = valid_config['plot_curves']
        self.clusters = valid_config['clusters']

        self.directories = valid_config['directories']
        self.anim = valid_config['anim']
        self.analyze = valid_config['analyze']

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser("Data analysis")

        parser.add_argument('--dirs', nargs='+', required=True, help='All the directories you want')
        parser.add_argument('--anim', dest='anim', action='store_true', help="Do you want animations?")
        parser.add_argument('--analyze', dest='analyze', action='store_true', help='Calculate metrics?')

        parser.add_argument('--detect', dest='detect', action='store_true', help='Generate metrics?')
        parser.add_argument('--train', dest='train', action='store_true', help='Analyze for training data?')
        parser.add_argument('--valid', dest='valid', action='store_true', help='Analyze for validation data?')
        parser.add_argument('--test', dest='test', action='store_true', help='Analyze for test data?')

        parser.add_argument('--config', default='validate/valid_config.json', help='json file')
        parser.add_argument('--clusters', default='clusters_v1.pkl', help='Name of the file to store summary statistics')

        args = parser.parse_args()

        try:
            with open(args.config, 'r') as f:
                valid_config = json.load(f)

            valid_config['directories'] = args.dirs
            valid_config['anim'] = args.anim
            valid_config['analyze'] = args.analyze

            # Over ride arguments in valid_config json file
            valid_config['detect'] = args.detect
            valid_config['train'] = args.train
            valid_config['valid'] = args.valid
            valid_config['test'] = args.test
            valid_config['clusters'] = args.clusters

            for k in valid_config.keys():
                print(k, valid_config[k])

        except FileNotFoundError:
            print("ERROR: Config file not found: {}".format(args.config))
            sys.exit(1)
        except json.JSONDecodeError:
            print("ERROR: Config file is not a valid JSON file!")
            sys.exit(1)

        return valid_config

    def analyze_main(self):

        directories = self.directories
        anim = self.anim
        analyze = self.analyze

        if analyze:
            processes = []
            for directory in directories:
                present_directory = os.path.join(self.root, directory)
                processes.append(
                    mp.Process(target=self.analyze_fun, args=(present_directory, None,)))

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        # Generate animations
        if anim:
            processes = []
            for directory in directories:
                present_directory = os.path.join(self.root, directory)
                processes.append(mp.Process(target=self.animate, args=(present_directory,)))

            for p in processes:
                p.start()

            for p in processes:
                p.join()


if __name__ == '__main__':
    # ================================
    # Parse Args stored in config file

    validate = Validate()
    validate.analyze_main()
