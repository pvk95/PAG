"""
Author: Varaha Karthik

The main function. For training, validation and testing.
"""
import sys
import torch

sys.path.append('./validate/')
import os
import json
import argparse
import sys
from api import Baselines
from api import PAG
import datetime
import copy


# ===========================================================================
# ================ Main entry point for all the algorithms ==================
# ===========================================================================

def log_config(config):
    """
    Writes the config file to the appropriate folder
    :param config: The config file for PAG:ct or PAG:ct+pet model
    :return: None
    """
    save_dir = config['exp_info']['save_dir']

    with open(os.path.join(save_dir, 'config_{}.txt'.format(config['exp_info']['exp_name'])), 'a+') as f:
        keys = config.keys()
        f.write('\n\n========' + str(datetime.datetime.now()) + '===========\n')
        f.write('===' + str(config['exp_info']['task']) + '===\n')
        for k in keys:
            if isinstance(config[k], dict):
                f.write('\n')
                for kk in config[k]:
                    f.writelines([kk, ' : ', str(config[k][kk]), '\n'])
            else:
                f.writelines([k, ' : ', str(config[k]), '\n'])


def biProcess(config):
    """
    Accepts the config file for PAG:ct+pet model for validation and testing purposes.
    :param config: Config dict for PAG:ct+pet model
    :return: None
    """
    if config is None:
        return None

    print("++++++++++++++++++++++++ PAG-ct+pet predictions ++++++++++++++++++++++++")
    assert config['exp_info']['method'] == 'pet_attn'

    ckpt_file = config['exp_info']['ckpt_file']
    save_dir = config['exp_info']['save_dir']
    exp_name = config['exp_info']['exp_name']
    state = torch.load(ckpt_file, map_location='cpu')
    train_logs = state['train_logs']
    new_state = {'train_logs': train_logs}

    torch.save(new_state, os.path.join(save_dir, f'model_{exp_name}.pt'))

    network = PAG(config=config)

    valid, test = False, False
    if config['exp_info']['task'] == "train" and config['exp_info']['n_folds'] != 1:
        valid = True
    elif config['exp_info']['task'] == "train" and config['exp_info']['n_folds'] == 1:
        test = True
    elif config['exp_info']['task'] == "valid":
        valid = True
    elif config['exp_info']['task'] == 'test':
        test = True
    elif config['exp_info']['task'] == 'valid+test':
        valid, test = True, True

    print(valid)
    print(test)
    print(config['exp_info']['task'])

    network.get_predictions(train=False, valid=valid, test=test, include_pet=True)


def uniProcess(config):
    """
     Accepts the config dict for PAG:ct model. Main function that starts the training, validation and testing.
     :param config: Config dict for PAG:ct model
     :return: None
     """

    if config['exp_info']['method'] == 'pet_attn':
        network = PAG(config=config)
    else:
        network = Baselines(config=config)

    if config['exp_info']['task'] == 'train':
        network.train()
        valid = True if config['exp_info']['n_folds'] != 1 else False
        print("++++++++++++++++++++++++ PAG-ct predictions ++++++++++++++++++++++++")
        network.get_predictions(train=False, valid=valid, test=not valid)
    else:

        valid = False
        test = False

        if config['exp_info']['task'] == "valid":
            valid = True
        elif config['exp_info']['task'] == 'test':
            test = True
        elif config['exp_info']['task'] == 'valid+test':
            valid, test = True, True
        else:
            raise ValueError
        print("++++++++++++++++++++++++ PAG-ct predictions ++++++++++++++++++++++++")
        network.get_predictions(train=False, valid=valid, test=test)


def AccomodatePAG(config):
    """
    PAG model encompasses two functions. PAG:ct and PAG:ct+pet
    :param config: The config for PAG is duplicated for PAG:ct+pet model and stored separately.
    :return:
    """
    method = config['exp_info']['method']
    save_dir = config['exp_info']['save_dir']
    exp_name = config['exp_info']['exp_name']

    if method != 'pet_attn':

        config['exp_info']['save_dir'] = os.path.join(save_dir, exp_name)

        if not os.path.exists(config['exp_info']['save_dir']):
            os.makedirs(config['exp_info']['save_dir'])

        if config['exp_info']['ckpt_file'] == 'None':
            config['exp_info']['ckpt_file'] = os.path.join(config['exp_info']['save_dir'],
                                                           'model_{}.pt'.format(config['exp_info']['exp_name']))

        log_config(config=config)

        return config, None
    else:

        uni_dir = os.path.join(save_dir, 'PAG-ct')
        bi_dir = os.path.join(save_dir, 'PAG-ct-pet')

        uni_config = copy.deepcopy(config)
        bi_config = copy.deepcopy(config)

        uni_config['exp_info']['save_dir'] = os.path.join(uni_dir, exp_name)
        bi_config['exp_info']['save_dir'] = os.path.join(bi_dir, exp_name)

        if not os.path.exists(uni_config['exp_info']['save_dir']):
            os.makedirs(uni_config['exp_info']['save_dir'])
        if not os.path.exists(bi_config['exp_info']['save_dir']):
            os.makedirs(bi_config['exp_info']['save_dir'])

        if uni_config['exp_info']['ckpt_file'] == 'None':
            uni_config['exp_info']['ckpt_file'] = os.path.join(uni_config['exp_info']['save_dir'],
                                                               'model_{}.pt'.format(uni_config['exp_info']['exp_name']))

        bi_config['exp_info']['ckpt_file'] = uni_config['exp_info']['ckpt_file']

        log_config(config=uni_config)
        log_config(config=bi_config)

        return uni_config, bi_config


def parse_args():
    """
    All the configurations are mentioned in the config file in JSON format
    Parse the config file and over-ride if necessary.
    :return:
    """

    # Create a parser
    parser = argparse.ArgumentParser('Segmentation of lung tumors PAG model and baselines')
    parser.add_argument('--config', help='Configuration file', type=str, default='config.json')

    parser.add_argument('--method', help='Which model?', type=str, default=None)

    parser.add_argument('--train', help="Train the model?", dest='train', action='store_true')
    parser.add_argument('--valid', help="Train the model?", dest='valid', action='store_true')
    parser.add_argument('--test', help="Train the model?", dest='test', action='store_true')

    parser.add_argument('--exp_name', help='Name of the experiment', default=None, type=str)
    parser.add_argument('--save_dir', help='Where do you want to save ?', default=None, type=str)
    parser.add_argument('--ckpt_file', help='Optional path for checkpoint', default=None, type=str)
    parser.add_argument('--fold', help='Fold no. in CV experiments', default=None, type=int)
    parser.add_argument('--n_folds', help='No. of folds', default=None, type=int)

    parser.add_argument('--n_epochs', help='Number of epochs', default=None, type=int)
    parser.add_argument('--lr', help='learning rate', default=None, type=float)

    # Parse the arguments
    args = parser.parse_args()

    # Try to open the file args.config
    try:
        with open(args.config, 'r') as config_file:
            config: dict = json.load(config_file)

            # ===== Exp Info =====
            if args.method is not None:
                config['exp_info']['method'] = args.method

            if args.train:
                config['exp_info']['task'] = "train"
            elif args.valid and not args.test:
                config['exp_info']['task'] = "valid"
            elif args.test and not args.valid:
                config['exp_info']['task'] = "test"
            elif args.valid and args.test:
                config['exp_info']['task'] = "valid+test"
            else:
                raise ValueError

            if args.exp_name is not None:
                config['exp_info']['exp_name'] = args.exp_name

            if args.save_dir is not None:
                config['exp_info']['save_dir'] = args.save_dir

            if args.ckpt_file is not None:
                config['exp_info']['ckpt_file'] = args.ckpt_file

            if args.fold is not None:
                config['exp_info']['fold'] = args.fold

            if args.n_folds is not None:
                config['exp_info']['n_folds'] = args.n_folds

            # ===== Hyper Params =====
            if args.n_epochs is not None:
                config['hyper_params']['n_epochs'] = args.n_epochs

            if args.lr is not None:
                config['hyper_params']['lr'] = args.lr

            uni_config, bi_config = AccomodatePAG(config=config)  # Duplicate config file for PAG:ct+pet model

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config))
        sys.exit(1)
    except json.JSONDecodeError:
        print("ERROR: Config file is not a valid JSON file!")
        sys.exit(1)

    return uni_config, bi_config


if __name__ == '__main__':

    # ================================
    # Parse Args stored in config file
    uni_config, bi_config = parse_args()
    for k in uni_config.keys():
        print(k, uni_config[k])

    uniProcess(config=uni_config)  # Train the PAG model or  baseline models. Valid/rest PAG:ct model
    biProcess(config=bi_config)  # Valid/Test PAG:ct+pet model
