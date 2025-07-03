import datetime
import logging
import argparse
import math
import time
import torch
from os import path as osp
import sys
sys.path.append("/mnt/petrelfs/liuyidi/Code/dreamUHD")
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.optionsval import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            # train_set = build_dataset(dataset_opt)
            # train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            # train_loader = build_dataloader(
            #     train_set,
            #     dataset_opt,
            #     num_gpu=1,
            #     dist=opt['dist'],
            #     sampler=train_sampler,
            #     seed=opt['manual_seed'])

            # num_iter_per_epoch = math.ceil(
            #     len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            # total_iters = int(opt['train']['total_iter'])
            # total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            # logger.info('Training statistics:'
            #             f'\n\tNumber of train images: {len(train_set)}'
            #             f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
            #             f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
            #             f'\n\tWorld size (gpu number): {opt["world_size"]}'
            #             f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            #             f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
            train_sampler =None
            total_epochs, total_iters = 0,0
            pass
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            # logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        # device_id = torch.cuda.current_device()
        # resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_state = torch.load(resume_state_path, map_location='cpu')
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):

    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    # resume_state = load_resume_state(opt)
    # # mkdir for experiments and logger
    # if resume_state is None:
    #     make_exp_dirs(opt)
    #     if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
    #         mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root


    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger=None)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    enhance_weight_path = args.weight
    # model.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
    model.load_network(model.net_g, args.weight)
    print(args.weight)
    # model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, -1, None, True)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
