"""
Author: Varaha Karthik (08 May 2020)

This file is the main API for training the models, saving the checkpoints, generating predictions.
The training strategy for the baseline models and the PAG model are different. However both of them share a similar
functions. Hence there are two classes for each one of them.

The file also includes other models other than the baseline models which were part of experiments.
They all share similar backbone architecture.
"""

import torch
import numpy as np
import h5py
import os

from torch.utils.tensorboard import SummaryWriter
import sys

from models.seg_model import Seg
from models.reg_model import Reg
from models.reg_seg_model import SegReg
from models.seg_pet_model import SegPET
from models.pet_model import PetModel
from models.pet_attention import SegAttnPet

from losses import Loss_functions
import multiprocessing as mp
from validate import Analyze
import random

LOAD_BATCH = 100  # No. of images you would want to load at once
N_TEST = 17  # No. of images as test time

# No. of epochs to train PAG model with CT images before you start random inclusion/exclusion of PET images.
BURN_IN_EPOCHS = 10

NETWORK_NAMES = ['ct', 'pet', 'mask', 'pet_mask']  # The names of the output branch. Disregard for baseline models.


# =======================================================================================
# ======== Baseline models ========
# ct: unimdoal model
# ct_both: bimodal model
# ct_attn: ct+attn model
# ct_both_attn: bimodal+attn model
# ======== PAG model ========
# pet_attn: PAG model

# ================
# Other models for other experiments

# =======================================================================================


def get_model(method='ct'):
    if method == 'reg':
        model = Reg()
    elif method == 'ct':
        model = Seg(both=False)
    elif method == 'ct_both':
        model = Seg(both=True)
    elif method == 'ct_reg':
        model = SegReg()
    elif method == 'ct_pet':
        model = SegPET()
    elif method == 'pet':
        model = PetModel()
    elif method == 'ct_attn':
        model = Seg(attention=True)
    elif method == 'ct_both_attn':
        model = Seg(both=True, attention=True)
    elif method == 'mask_basic' or method == 'mask_prf':
        model = SegPET(mask=method)
    elif method == 'pet_attn':
        model = SegAttnPet()
    else:
        print("Improper method")
        raise ValueError

    return model


class AE(Loss_functions):
    _patience = 15
    _epsilon = 0.01
    # _methods = ['reg', 'ct', 'ct_both', 'ct_reg', 'ct_pet', 'pet', 'attn_ct', 'ct_both_attn',
    #            'mask_basic', 'mask_prf']
    _methods = ['ct', 'ct_both', 'ct_pet', 'mask_basic', 'pet', 'ct_attn', 'pet_attn', 'ct_both_attn']

    def __init__(self, config):
        """
        :param config: The configurations as defined in the configurations file
        """

        super(Loss_functions).__init__()

        self.seed = config['seed']
        # Required information for data
        self.data_file = config['data_info']['data_file']
        self.data_aug = config['data_info']['data_aug']

        # Required hyper parameters
        self.n_epochs = config['hyper_params']['n_epochs']
        self.lr = config['hyper_params']['lr']
        self.batch_size = config['hyper_params']['batch_size']

        self.alpha = config['hyper_params']['alpha']
        self.alpha_dice = config['hyper_params']['alpha_dice']
        self.beta = config['hyper_params']['beta']
        self.gamma = config['hyper_params']['gamma']
        self.weight_decay = config['hyper_params']['weight_decay']
        self.n_train_valid = config['hyper_params']['n_train_valid']
        self.frac_pet = config['hyper_params']['frac_pet']  # All are ensured that they are included in training
        self.aug = config['hyper_params']['data_aug']
        self.include_pet_prob = config['hyper_params']['include_pet_prob']

        # Required experiment information
        self.method = config['exp_info']['method']
        try:
            assert self.method in AE._methods
        except AssertionError as e:
            print(e)
            print("Improper method given as input. Recheck your method")
            print("Allowed Values: ", AE._methods)
            sys.exit(1)
        self.exp_name = config['exp_info']['exp_name']
        self.save_dir = config['exp_info']['save_dir']
        self.ckpt_file = config['exp_info']['ckpt_file']
        self.fold = config['exp_info']['fold']
        self.n_folds = config['exp_info']['n_folds']
        self.start_epoch = config['exp_info']['start_epoch']

        # Build the model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(method=self.method).to(self.device)

        # For training the model
        self.train_logs = None  # Contains all the information of training. Loaded/Initialized from self.laod_model()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)  # Optimizer

        # self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([500.0]).to(self.device))
        self.bce_loss = torch.nn.BCELoss()  # BCE loss
        self.loss_fun = None  # Composite loss (Refer thesis) function for basleine models and PAG model.
        self.init_loss_fun()

        self.load_model()
        # self.load_pre_model()

    def init_loss_fun(self):
        """
        Initializes suitable loss function for appropriate model
        """
        if self.method == 'reg':
            self.loss_fun = self.loss_reg

        elif self.method == 'ct' or self.method == 'ct_both' or \
                self.method == 'ct_attn' or self.method == 'ct_both_attn' or \
                self.method == 'pet_attn':
            self.loss_fun = self.loss_ct

        elif self.method == 'ct_reg':
            self.loss_fun = self.loss_ct_reg

        elif self.method == 'ct_pet':
            self.loss_fun = self.loss_ct_pet

        elif self.method == 'pet':
            self.loss_fun = self.loss_pet
        elif self.method == 'mask_basic' or self.method == 'mask_prf':
            self.loss_fun = self.loss_mask
        else:
            print("Improper method")
            raise ValueError

    # ===================================
    # ======= Save and Load models ======
    # ===================================

    def save_model(self, train_losses, valid_losses, epoch):
        """
        Save checkpoints
        :param train_losses: The epoch wise loss for training data
        :param valid_losses: The epoch wise loss for valid/test data
        :param epoch: Epoch number
        :return:
        """

        losses = {'train_losses': train_losses, 'valid_losses': valid_losses}
        self.train_logs['losses'] = losses

        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_logs': self.train_logs}

        filename = os.path.join(self.save_dir, f'model_{self.exp_name}.pt')

        torch.save(state, filename)
        print("Saved model to {}".format(self.ckpt_file))

    def init_losses(self):
        """
        Initializes the loss functions
        """

        train_losses = {'Combined': [], 'Dice': [], 'CCE': []}
        valid_losses = {'Combined': [], 'Dice': [], 'CCE': []}
        if self.method == 'ct_pet' or self.method == 'pet':
            train_losses['PET_Recon'] = []
            valid_losses['PET_Recon'] = []
        elif self.method == 'mask_basic':
            train_losses['PET_Recon'] = []
            valid_losses['PET_Recon'] = []

            train_losses['Seg_Pet'] = []
            valid_losses['Seg_Pet'] = []
        elif self.method == 'pet_attn':
            valid_losses['Combined_PET'] = []
            valid_losses['Dice_PET'] = []
            valid_losses['CCE_PET'] = []

        return train_losses, valid_losses

    def load_model(self):
        """
        Load the model from checkpoint. Looks for checkpoint in the self.save_dir (passed from config file)
        :return:
        """

        look_path = self.ckpt_file
        if os.path.isfile(look_path):
            print(f"Found a checkpoint. Loading model from {look_path}")
            state = torch.load(look_path, map_location=self.device)

            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()
            self.optimizer.load_state_dict(state['optimizer'])
            self.train_logs = state['train_logs']

            print("Loaded model successfully")

        else:
            train_idxs, valid_idxs, test_idxs, pet_idxs = self.train_idxs()
            train_losses, valid_losses = self.init_losses()

            self.train_logs = {'losses': {'train_losses': train_losses, 'valid_losses': valid_losses},
                               'train_idxs': train_idxs,
                               'valid_idxs': valid_idxs,
                               'test_idxs': test_idxs,
                               'pet_idxs': pet_idxs}

    # ===================================
    # ============ Inference ============
    # ===================================

    def patch_extra_data(self, data_type, extra_indexes, include_pet, predict_file_name):
        """
        Initially we considered only 17 test images. Later, we included 60 more images.
        :param data_type: 'test' only
        :param extra_indexes: Indices of extra data
        :param include_pet: Whether to include PET images or not. Appropriate for PAG model only
        :param predict_file_name: Name of predictions.
        :return:
        """

        try:
            assert data_type == 'test'
        except AssertionError:
            print("Extra indexes only for test data. Not for valid or train")

        try:
            assert len(extra_indexes) < LOAD_BATCH
        except AssertionError:
            print("Extra indexes larger than feasible!")
            sys.exit(0)

        n_out = len(NETWORK_NAMES)

        print("Generating extra predictions for {} data".format(data_type))

        file = h5py.File(self.data_file, 'r')

        img_params = file['data/extra_data'].shape[1:5]

        network_preds = [False] * n_out

        # Initialize predictions with zeros.
        extra_predictions = np.zeros(shape=(len(extra_indexes),) + img_params + (n_out,), dtype=np.float32)
        # Generate predictions sequentially. Batch-wise not implemented.
        with torch.no_grad():
            data = file['data/extra_data'][()]
            set_samples = data.shape[0]

            for data_idx in np.arange(set_samples):

                print(f"\nImage index:  {extra_indexes[data_idx]}", end=" ")
                inputs = self.get_inputs(data[data_idx:data_idx + 1, ...])

                if self.method == 'ct_both' or self.method == 'ct_both_attn':
                    outputs = self.model(torch.cat((inputs[0], inputs[1]), dim=1))
                elif self.method == 'pet_attn' and include_pet:

                    print(" (Included PET) ", end=' ')
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs[0])

                assert len(outputs) == n_out
                for j in range(n_out):
                    if outputs[j] is not None:
                        network_preds[j] = True
                        extra_predictions[data_idx, :, :, :, :, j] = outputs[j].squeeze(0).cpu().numpy()
            print("\n")

        with h5py.File(os.path.join(self.save_dir, predict_file_name), 'a') as file:
            # for n in network_names:
            #   file.create_group(name='extra/{}'.format(n))

            for i in range(len(network_preds)):
                if network_preds[i]:
                    try:
                        file.create_dataset(name='extra/{}/{}'.format(NETWORK_NAMES[i], data_type),
                                            data=extra_predictions[..., i], dtype=np.float32)
                    except RuntimeError:
                        print('extra/{}/{} exists. Skipping (Discarding predictions)...'.
                              format(NETWORK_NAMES[i], data_type))

    def predict(self, data_type, data_indexes, predict_file_name='./seg_predictions.h5',
                include_pet=False, extra_indexes=None, mode_h5='w'):
        """
        Perform predictions on appropriate data.
        :param data_type: ['train', 'valid', 'test']
        :param data_indexes: The indices of the data you want to predict
        :param predict_file_name: Name of prediction file
        :param include_pet: Whether to include PET images or not.
        :param extra_indexes: The extra indices of extra test data.
        :param mode_h5: Mode of writing.
        :return:
        """

        n_samples = len(data_indexes)

        n_out = len(NETWORK_NAMES)

        print("Generating predictions for {} data".format(data_type))

        file = h5py.File(self.data_file, 'r')

        img_params = file['data/data'].shape[1:5]
        shape = (n_samples,) + img_params
        predictions = np.zeros(shape=shape + (n_out,), dtype=np.float32)

        network_preds = [False] * n_out

        with torch.no_grad():
            data = file['data/data'][data_indexes, ...]
            for data_idx in np.arange(n_samples):
                print(f"\nImage index:  {data_indexes[data_idx]}", end=" ")
                inputs = self.get_inputs(data[data_idx:data_idx + 1, ...])

                if self.method == 'ct_both' or self.method == 'ct_both_attn':
                    outputs = self.model(torch.cat((inputs[0], inputs[1]), dim=1))
                elif self.method == 'pet_attn' and include_pet:
                    print(" (Included PET) ", end=' ')
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs[0])

                assert len(outputs) == n_out
                for j in range(n_out):
                    if outputs[j] is not None:
                        network_preds[j] = True
                        predictions[data_idx, :, :, :, :, j] = outputs[j].squeeze(0).cpu().numpy()
        print("\n")
        file.close()

        if mode_h5 == 'w':
            print("Over-riding any existing predictions to {}".format(os.path.join(self.save_dir, predict_file_name)))

        with h5py.File(os.path.join(self.save_dir, predict_file_name), mode=mode_h5) as file:

            for i in range(len(network_preds)):
                if network_preds[i]:
                    try:
                        file.create_dataset(name='{}/{}'.format(NETWORK_NAMES[i], data_type),
                                            data=predictions[..., i], dtype=np.float32)
                    except RuntimeError:
                        print('{}/{} exists. Skipping (Discarding predictions)...'.
                              format(NETWORK_NAMES[i], data_type))

        if extra_indexes is not None:
            self.patch_extra_data(data_type=data_type, extra_indexes=extra_indexes,
                                  include_pet=include_pet, predict_file_name=predict_file_name)

    def get_predictions(self, train=False, valid=True, test=True, include_pet=False,
                        predict_file_name='./seg_predictions.h5'):

        """
        Calls the predict function. API for predictions.
        :param train: Train predictions?
        :param valid: Valid predictions?
        :param test: Test predictions?
        :param include_pet: Whether to include PET images ? Suitable for PAG model
        :param predict_file_name: Name of predictions
        :return:
        """

        print("Over-riding any existing predictions to {}".format(os.path.join(self.save_dir, predict_file_name)))
        with h5py.File(os.path.join(self.save_dir, predict_file_name), 'w') as file:
            for n in NETWORK_NAMES:
                file.create_group(name=n)

        train_curves = self.train_logs

        train_idxs = train_curves['train_idxs']
        valid_idxs = train_curves['valid_idxs']
        test_idxs = train_curves['test_idxs']

        if train:
            self.predict(data_type='train', data_indexes=train_idxs)
        if valid:
            self.predict(data_type='valid', data_indexes=valid_idxs, include_pet=include_pet, mode_h5='a')
        if test:
            self.predict(data_type='test', data_indexes=test_idxs, include_pet=include_pet,
                         extra_indexes=np.arange(337, 397), mode_h5='a')

    # ===================================
    # ============ Validation ===========
    # ===================================

    def validate_metric(self, epoch, writer):
        """
        Calculate metrics of interest while training
        :param epoch: Current epoch
        :param writer: The tenorboard writer
        :return:
        """

        if self.n_folds == 1:
            valid = False
        else:
            valid = True

        valid_config = {
            "root": "/home/karthikp/MST/AE/",
            "data_path": "/home/karthikp/MST/tumor_data/",
            "tags_consider": ["tumor-tag", "node-tag", "mets-tag", "all", "all_dice"],
            "detect": True,
            "train": False,
            "valid": valid,
            "test": not valid,
            "plot_curves": False}

        parent, child = mp.Pipe(duplex=False)

        data_type = 'valid' if valid else 'test'
        data_indexes = self.train_logs['valid_idxs'] if valid else self.train_logs['test_idxs']

        if self.method == 'pet_attn':

            new_state = {'train_logs': self.train_logs}
            to_save_dir = os.path.join(self.save_dir, os.path.join('../../PAG-ct-pet/', self.exp_name))
            torch.save(new_state, os.path.join(to_save_dir, f'model_{self.exp_name}.pt'))
            self.predict(data_type=data_type, data_indexes=data_indexes,
                         predict_file_name=os.path.join('../../PAG-ct-pet/',
                                                        self.exp_name, 'seg_predictions.h5'), include_pet=True)

        else:
            to_save_dir = self.save_dir

            self.predict(data_type=data_type, data_indexes=data_indexes)

        ae = Analyze(valid_config=valid_config)
        ae.detect_fn(save_dir=to_save_dir)

        try:
            ae.calculate_metrics(save_dir=to_save_dir, conn=child)

            metrics, _ = parent.recv()
            print(metrics)

            writer.add_scalar('Metrics/sens-tumor', metrics[0][0], global_step=epoch)
            writer.add_scalar('Metrics/sens-node', metrics[1][0], global_step=epoch)
            writer.add_scalar('Metrics/sens-mets', metrics[2][0], global_step=epoch)
            writer.add_scalar('Metrics/auprc-tumor', metrics[0][1], global_step=epoch)
            writer.add_scalar('Metrics/auprc-node', metrics[1][1], global_step=epoch)
            writer.add_scalar('Metrics/auprc-mets', metrics[2][1], global_step=epoch)

        except ValueError as e:
            print(e)
            pass
        except:
            pass

    def validate_model(self, valid_idxs, include_pet=False):

        """
        Perform validation during the course of training.
        Predictions of validation data. After every epoch.
        :param valid_idxs: The indices of validation data.
        :param include_pet: Whether to include PET images. Suitable for PAG model
        :return:
        """

        if self.n_folds != 1:
            data = self.load_data(valid_idxs=valid_idxs)
        else:
            test_idxs = self.train_logs['test_idxs']
            data = self.load_data(test_idxs=test_idxs)

        losses = {'Combined': 0}

        n_samples = data.shape[0]

        with torch.no_grad():
            for data_idx in np.arange(n_samples):
                img = data[data_idx:data_idx + 1, ...]

                inputs = self.get_inputs(img)
                if self.method == 'ct_both' or self.method == 'ct_both_attn':
                    outputs = self.model(torch.cat((inputs[0], inputs[1]), dim=1))
                elif self.method == 'pet_attn' and include_pet:
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs[0])
                total_loss, ind_loss = self.loss_fun(outputs=outputs, inputs=inputs)

                losses['Combined'] += total_loss.item()
                for k in ind_loss.keys():
                    if k not in losses.keys():
                        losses[k] = ind_loss[k]
                    else:
                        losses[k] += ind_loss[k]

        for k in losses.keys():
            losses[k] /= n_samples

        return losses

    # ===================================
    # ============ Training =============
    # ===================================

    def lr_scheduler(self, epoch):
        """
        Decay the learning rate
        :param epoch: Epoch
        :return: Decayed LR
        """

        return self.lr * ((1 - epoch / self.n_epochs) ** 0.9)

    def train_idxs(self):
        """
        Randomly select the training, validation and test indices.
        1. First shuffle the indices
        2. Then select N_TEST indices
        3.  The rest is for training and validation
        4. Four fold CV
        5. Sample n_pet images when reduced.
        :return: Appropriate indices
        """

        with h5py.File(self.data_file, 'r') as file:
            total_samples = file['data/data'].shape[0]

        np.random.seed(self.seed)
        sample_idxs = np.arange(total_samples)
        np.random.shuffle(sample_idxs)

        # test_idxs = sample_idxs[:self.n_test]
        # train_valid = sample_idxs[self.n_test:]

        test_idxs = sample_idxs[:N_TEST]
        train_valid = np.random.choice(sample_idxs[N_TEST:], replace=False, size=self.n_train_valid)

        if self.n_folds == 1:
            train_idxs = train_valid
            valid_idxs = []
        else:
            smp_fold = len(train_valid) // self.n_folds

            valid_idxs = train_valid[self.fold * smp_fold:self.fold * smp_fold + smp_fold]
            train_idxs = [i for i in train_valid if i not in valid_idxs]

        if self.method == 'pet_attn' and self.n_folds == 1 and self.frac_pet == 0.03:
            n_pet = np.ceil(self.frac_pet * len(train_idxs)).astype(np.int)
            pet_idxs = np.random.choice(train_idxs, size=n_pet, replace=False)
        else:
            n_pet = np.floor(self.frac_pet * len(train_idxs)).astype(np.int)
            pet_idxs = np.random.choice(train_idxs, size=n_pet, replace=False)

        if self.method != 'pet_attn':
            train_idxs = np.copy(pet_idxs)

        return np.sort(train_idxs), np.sort(valid_idxs), np.sort(test_idxs), np.sort(pet_idxs)

    def load_data(self, valid_idxs=None, test_idxs=None):
        """
        Load validation/test data
        """

        data = None
        if valid_idxs is not None:
            with h5py.File(self.data_file, 'r') as file:
                data = file['data/data'][valid_idxs, ...]

        elif test_idxs is not None:
            with h5py.File(self.data_file, 'r') as file:
                data = file['data/data'][test_idxs, ...]

        else:
            Warning("No data loaded")

        return data

    def get_inputs(self, img):
        """
        Transfer the image to Pytorch Tensors and transfer them to the GPU.
        :param img: The input image
        :return: Appropriate tensors
        """

        img_tensor = torch.from_numpy(img[..., 0]).float().to(self.device)
        pet_tensor = torch.from_numpy(img[..., 1]).float().to(self.device)
        seg_mask = torch.from_numpy(img[..., 2]).float().to(self.device)

        return img_tensor, pet_tensor, seg_mask

    @staticmethod
    def log_data(writer, train_losses, valid_losses, epoch):
        """
        Log the training and validation curves in the tensorboard
        :param writer: Tensorboard writer
        :param train_losses: The training indices
        :param valid_losses: The validation/test indices where appropriate.
        :param epoch: Current epoch
        :return:
        """

        print(f"========")
        print("Mean training loss: {}".format(round(train_losses['Combined'][-1], 5)))
        for k in train_losses.keys():
            writer.add_scalar(f'Train/{k}', train_losses[k][-1], global_step=epoch)

        for k in valid_losses.keys():
            writer.add_scalar(f'Valid/{k}', valid_losses[k][-1], global_step=epoch)

        print("Mean validation loss: {}".format(round(valid_losses['Combined'][-1], 5)))


# ==========================================
# ============ Baseline class  =============
# ==========================================

class Baselines(AE):
    """
    This class implemented baseline mdoels specific functions. Inherits all teh fucnctions from the AE class
    """

    def __init__(self, config):

        super().__init__(config=config)

        try:
            assert self.method != 'pet_attn'
        except AssertionError:
            print("method not a baseline method")
            print("Baseline class is not for PAG model.")

    def load_train_data(self, train_idxs, set_idx=0, aug=False):

        """
        Training data is broken into sets for every epoch. Each set consists a maximum of LOAD_BATCH samples.
        Accordingly, load that particular set.
        :param train_idxs: Training indices
        :param set_idx: The index of the set
        :param aug: Whether augmented data or true data?
        :return: The training data
        """

        data_file = self.data_file if not aug else self.data_aug

        print(data_file)

        if len(train_idxs) <= LOAD_BATCH:
            with h5py.File(data_file, 'r') as file:
                data = file[f'data/data'][np.sort(train_idxs), ...]
        else:
            tr_idxs = train_idxs[set_idx * LOAD_BATCH: set_idx * LOAD_BATCH + LOAD_BATCH]
            with h5py.File(data_file, 'r') as file:
                data = file[f'data/data'][np.sort(tr_idxs), ...]

        return data

    def step_train(self, iteration, batch_size, train_data):
        """
        One step of training.
        :param iteration: The iteration nunber
        :param batch_size: Batch size
        :param train_data: the training data
        :return:
        """

        begin = iteration * batch_size
        end = begin + batch_size

        img = train_data[begin:end, ...]  # Load the images

        inputs = self.get_inputs(img)  # convert to image tensors
        # Generate outputs
        if self.method == 'ct_both' or self.method == 'ct_both_attn':
            outputs = self.model(torch.cat((inputs[0], inputs[1]), dim=1))
        else:
            outputs = self.model(inputs[0])
        total_loss, ind_loss = self.loss_fun(outputs, inputs)  # calculate the loss

        self.optimizer.zero_grad()  # Zero all the accumulated gradients
        total_loss.backward()  # Calculte the gradients
        self.optimizer.step()  # Backprop the gradients.

        losses = {'Combined': total_loss.item()}

        for k in ind_loss:
            losses[k] = ind_loss[k]

        del img

        return losses

    def train(self):
        """
        Train the models.
        1. Load the indices
        2. Calculate the number of sets
        3. Randomly shuffle the sets and allocate indices randomly to the sets
        4. For every epoch determine whether true data or augment data with a probability
        5. Load approapriate data and make a step
        6. Store the losses
        :return:
        """

        train_logs = self.train_logs

        train_idxs = train_logs['train_idxs']
        valid_idxs = train_logs['valid_idxs']

        train_losses = train_logs['losses']['train_losses']
        valid_losses = train_logs['losses']['valid_losses']

        n_sets = np.ceil(len(train_idxs) / LOAD_BATCH).astype(np.int)
        train_sets = np.arange(n_sets)

        batch_size = self.batch_size
        writer = SummaryWriter(log_dir=self.save_dir)

        for epoch in range(self.start_epoch, self.n_epochs):

            print(f"========= Epoch No: {epoch} =========")

            iter_losses = {}
            for k in train_losses.keys():
                iter_losses[k] = []

            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_scheduler(epoch)

            np.random.shuffle(train_sets)
            np.random.shuffle(train_idxs)
            step_counter = 0

            val = np.random.uniform(0, 1)

            if val < self.aug:
                n_iter_set = 2
            else:
                n_iter_set = 1
            print(val)

            for set_idx in train_sets:
                do_aug = False
                for _ in range(n_iter_set):

                    train_data = self.load_train_data(train_idxs=train_idxs, set_idx=set_idx, aug=do_aug)
                    n_batch = train_data.shape[0]
                    np.random.shuffle(train_data)
                    num_iters = np.ceil(n_batch // batch_size).astype(np.int)

                    for it in range(num_iters):
                        step_losses = self.step_train(iteration=it, batch_size=batch_size,
                                                      train_data=train_data, epoch=epoch)

                        for k in iter_losses:
                            iter_losses[k].append(step_losses[k])
                        print("Training Loss at step {}: {}".format(step_counter, round(step_losses['Combined'], 5)))
                        step_counter += 1

                    del train_data
                    do_aug = not do_aug

            for k in train_losses.keys():
                train_losses[k].append(np.mean(iter_losses[k]))

            valid_iter_loss = self.validate_model(valid_idxs=valid_idxs)
            for k in valid_iter_loss.keys():
                valid_losses[k].append(valid_iter_loss[k])

            self.log_data(writer=writer, train_losses=train_losses, valid_losses=valid_losses, epoch=epoch)
            self.save_model(train_losses=train_losses, valid_losses=valid_losses, epoch=epoch)

            if (epoch + 1) % 5 == 0:
                self.validate_metric(epoch=epoch, writer=writer)


class PAG(AE):
    """
    The class for training the PAG model.
    """

    def __init__(self, config):

        super().__init__(config=config)

        try:
            assert self.method == 'pet_attn'
        except AssertionError:
            print("method not set to: (pet_attn)")
            print("PAG class is only for PAG model.")

    def get_set_idxs(self, ct_idxs, pet_idxs):
        """
        Ensure for every set that the ratio of PET/CT is maintained.
        :param ct_idxs: The CT indices for which PET images are missing. (Simulate)
        :param pet_idxs: The PET/CT indices.
        :return:
        """

        n_train = len(ct_idxs) + len(pet_idxs)

        n_sets = np.ceil(n_train / LOAD_BATCH).astype(np.int)

        train_sets = []

        np.random.shuffle(ct_idxs)
        np.random.shuffle(pet_idxs)

        for set_idx in np.arange(n_sets):
            fraction = (1 - self.frac_pet)
            n_select = np.int(fraction * LOAD_BATCH)
            ct_tr_idxs = np.sort(ct_idxs[set_idx * n_select: set_idx * n_select + n_select])

            fraction = self.frac_pet
            n_select = np.int(fraction * LOAD_BATCH)
            pet_tr_idxs = np.sort(pet_idxs[set_idx * n_select: set_idx * n_select + n_select])

            train_sets.append((ct_tr_idxs, pet_tr_idxs))

        return train_sets

    def load_train_data(self, ct_tr_idxs, pet_tr_idxs, aug=False):
        """
        Load appropraite data
        :param ct_tr_idxs: The training indices for CT images
        :param pet_tr_idxs: The training indices for PET/CT images
        :param aug: Whether to perform augmentation or not.
        :return:
        """

        data_file = self.data_file if not aug else self.data_aug

        print(data_file)
        n_imgs = len(ct_tr_idxs) + len(pet_tr_idxs)

        assert n_imgs <= LOAD_BATCH

        with h5py.File(data_file, 'r') as file:
            img_shape = file['data/data'].shape[1:]
            data = np.zeros(shape=((n_imgs,) + img_shape), dtype=np.float32)
            if not len(ct_tr_idxs) == 0:
                data[:len(ct_tr_idxs), ...] = file['data/data'][ct_tr_idxs, ...]
            data[len(ct_tr_idxs):, ...] = file['data/data'][pet_tr_idxs, ...]

        return data

    def step_train(self, ct_tr_idxs, pet_tr_idxs, train_data, epoch, step_counter):
        """
        1. Determine whether to include PET images or not.
        2. Sample PET indices and CT indices.
        3. Include PET indices if yes include_pet do not include.
        """

        include_pet = True if random.random() < self.include_pet_prob and epoch > BURN_IN_EPOCHS else False

        if include_pet:
            n_ct = len(ct_tr_idxs)
            try:
                choose_pos_idxs = np.random.choice(np.arange(n_ct, len(pet_tr_idxs) + n_ct), size=self.batch_size,
                                                   replace=False)
            except ValueError:
                print("Batch size larger. Sampling only one sample!")
                choose_pos_idxs = np.random.choice(np.arange(n_ct, len(pet_tr_idxs) + n_ct), size=1, replace=False)

            batch_idxs = pet_tr_idxs[choose_pos_idxs - n_ct]
            img = train_data[choose_pos_idxs, ...]
        else:
            choose_pos_idxs = np.random.choice(np.arange(len(ct_tr_idxs) + len(pet_tr_idxs)), size=self.batch_size,
                                               replace=False)
            img = train_data[choose_pos_idxs, ...]
            batch_idxs = np.concatenate((ct_tr_idxs, pet_tr_idxs))[choose_pos_idxs]

        inputs = self.get_inputs(img)

        if include_pet:
            print("Included PET at {}/{}".format(step_counter, epoch), end=" ")
            outputs = self.model(inputs[0], inputs[1])
        else:
            outputs = self.model(inputs[0])
        total_loss, ind_loss = self.loss_fun(outputs, inputs)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        losses = {'Combined': total_loss.item()}

        for k in ind_loss:
            losses[k] = ind_loss[k]

        print("Training Loss at step {} ({}): {}".format(step_counter, batch_idxs,
                                                         round(losses['Combined'], 5)))

        del img

        return losses

    def train(self):

        train_logs = self.train_logs

        train_idxs = train_logs['train_idxs']
        valid_idxs = train_logs['valid_idxs']
        pet_idxs = train_logs['pet_idxs']

        train_losses = train_logs['losses']['train_losses']
        valid_losses = train_logs['losses']['valid_losses']

        ct_idxs = np.array([i for i in train_idxs if i not in pet_idxs])

        writer = SummaryWriter(log_dir=self.save_dir)

        for epoch in range(self.start_epoch, self.n_epochs):

            print(f"========= Epoch No: {epoch} =========")

            iter_losses = {}
            for k in train_losses.keys():
                iter_losses[k] = []

            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_scheduler(epoch)

            step_counter = 0

            val = np.random.uniform(0, 1)

            if val < self.aug:
                n_iter_set = 2
            else:
                n_iter_set = 1
            print(val)

            train_sets = self.get_set_idxs(ct_idxs=ct_idxs, pet_idxs=pet_idxs)

            print(train_sets)

            random.shuffle(train_sets)

            for consider_idxs in train_sets:

                ct_tr_idxs, pet_tr_idxs = consider_idxs

                do_aug = False
                for _ in range(n_iter_set):

                    train_data = self.load_train_data(ct_tr_idxs, pet_tr_idxs, aug=do_aug)
                    n_batch = train_data.shape[0]

                    num_iters = np.ceil(n_batch // self.batch_size).astype(np.int)

                    for it in range(num_iters):
                        step_losses = self.step_train(ct_tr_idxs=ct_tr_idxs,
                                                      pet_tr_idxs=pet_tr_idxs,
                                                      train_data=train_data,
                                                      epoch=epoch, step_counter=step_counter)

                        for k in iter_losses:
                            iter_losses[k].append(step_losses[k])
                        step_counter += 1

                    del train_data
                    do_aug = not do_aug

            for k in train_losses.keys():
                train_losses[k].append(np.mean(iter_losses[k]))

            valid_iter_loss = self.validate_model(valid_idxs=valid_idxs)
            for k in valid_iter_loss.keys():
                valid_losses[k].append(valid_iter_loss[k])

            valid_iter_loss_PET = self.validate_model(valid_idxs=valid_idxs, include_pet=True)
            for k in valid_iter_loss_PET.keys():
                valid_losses[k + '_PET'].append(valid_iter_loss_PET[k])

            self.log_data(writer=writer, train_losses=train_losses, valid_losses=valid_losses, epoch=epoch)
            self.save_model(train_losses=train_losses, valid_losses=valid_losses, epoch=epoch)

            if (epoch + 1) % 5 == 0:
                self.validate_metric(epoch=epoch, writer=writer)
