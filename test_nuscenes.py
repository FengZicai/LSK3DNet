import os
import random
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import shutil
from easydict import EasyDict
from tqdm import tqdm

import yaml
from utils.load_util import load_yaml
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name, get_pc_model_class
from dataloader.dataset2 import get_dataset_class, get_collate_class
from network.largekernel_model import get_model_class

from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.erk_sparse_core import Masking, CosineDecay

import warnings
warnings.filterwarnings("ignore")

# Testing settings
parser = argparse.ArgumentParser(description='LSKNet Testing')
parser.add_argument('--config_path', default='./config/lk-nuscenes_erk_tta.yaml')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--port', default='3023', type=str)
parser.add_argument('--num_vote', type=int, default=28, help='number of voting in the test') #14 28
args = parser.parse_args()
config_path = args.config_path
configs = load_yaml(config_path)
configs.update(vars(args))  # override the configuration using the value in args
configs = EasyDict(configs)

configs['dataset_params']['val_data_loader']["batch_size"] = configs.num_vote
configs['dataset_params']['val_data_loader']["num_workers"] = 14 #configs.num_vote
if configs.num_vote > 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = True
    configs['dataset_params']['val_data_loader']["flip_aug"] = True
    configs['dataset_params']['val_data_loader']["scale_aug"] = True
    configs['dataset_params']['val_data_loader']["transform_aug"] = True
elif configs.num_vote == 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = False
    configs['dataset_params']['val_data_loader']["flip_aug"] = False
    configs['dataset_params']['val_data_loader']["scale_aug"] = False
    configs['dataset_params']['val_data_loader']["transform_aug"] = False

exp_dir_root = configs['model_params']['model_load_path'].split('/')
exp_dir_root = exp_dir_root[0] if len(exp_dir_root) > 1 else ''
exp_dir = './'+ exp_dir_root +'/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
shutil.copy('test_nuscenes.py', str(exp_dir))
shutil.copy('config/lk-nuscenes_erk_tta.yaml', str(exp_dir))


def main(configs):
    configs.nprocs = torch.cuda.device_count()
    configs.train_params.distributed = True if configs.nprocs > 1 else False
    if configs.train_params.distributed:
        mp.spawn(main_worker, nprocs=configs.nprocs, args=(configs.nprocs, configs))
    else:
        main_worker(0, 1, configs)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def main_worker(local_rank, nprocs, configs):
    torch.autograd.set_detect_anomaly(True)

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    train_hypers.local_rank = local_rank
    train_hypers.world_size = nprocs
    configs.train_params.world_size = nprocs
    
    if train_hypers['distributed']:
        init_method = 'tcp://' + args.ip + ':' + args.port
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=nprocs, rank=local_rank)
        dataset_config.val_data_loader.batch_size = dataset_config.val_data_loader.batch_size // nprocs

    pytorch_device = torch.device('cuda:' + str(local_rank))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)

    seed = train_hypers.seed + local_rank * dataset_config.val_data_loader.num_workers * train_hypers['max_num_epochs']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = get_model_class(model_config['model_architecture'])(configs)

    if train_hypers['distributed']:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    if os.path.exists(model_config['model_load_path']):
        print('pre-train')
        try:
            my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
        except:
            my_model = load_checkpoint_old(model_config['model_load_path'], my_model)

    my_model.to(pytorch_device)

    if train_hypers['distributed']:
        train_hypers.local_rank = train_hypers.local_rank % torch.cuda.device_count()
        my_model= DistributedDataParallel(my_model,device_ids=[train_hypers.local_rank],find_unused_parameters=False)


    # prepare dataset
    val_dataloader_config = dataset_config['val_data_loader']
    data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]

    label_mapping = dataset_config["label_mapping"]


    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    val_pt_dataset = SemKITTI(dataset_config, data_path, imageset=val_imageset, num_vote = configs.num_vote)

    val_dataset = get_dataset_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        config=dataset_config,
        loader_config=val_dataloader_config,
        num_vote = configs.num_vote)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=train_hypers.world_size, rank=train_hypers.local_rank, shuffle=False)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=val_dataloader_config["batch_size"],
                                                    collate_fn=get_collate_class(dataset_config['collate_type']),
                                                    num_workers=val_dataloader_config["num_workers"],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    shuffle = False,
                                                    sampler=val_sampler)


    if val_imageset == 'val':
        if train_hypers.local_rank == 0:
            # validation
            print('*'*80)
            print('Test network performance on validation split')
            print('*'*80)
            pbar = tqdm(total=len(val_dataset_loader), ncols=80)
        else:
            pbar = None

        my_model.eval()
        hist_list = []
        time_list = []
        with torch.no_grad():
            for i_iter_val, (val_data_dict) in enumerate(val_dataset_loader):
                    
                torch.cuda.empty_cache()
                raw_labels = val_data_dict['raw_labels'].to(pytorch_device)
                vote_logits = torch.zeros(raw_labels.shape[0], model_config['num_classes']).to(pytorch_device)
                indices = val_data_dict['indices'].to(pytorch_device)

                val_data_dict['points'] = val_data_dict['points'].to(pytorch_device)
                val_data_dict['normal'] = val_data_dict['normal'].to(pytorch_device)
                val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(pytorch_device)
                val_data_dict['labels'] = val_data_dict['labels'].to(pytorch_device)

                torch.cuda.synchronize()
                start_time = time.time()
                val_data_dict = my_model(val_data_dict)
                torch.cuda.synchronize()
                time_list.append(time.time()-start_time)
                logits = val_data_dict['logits']
                vote_logits.index_add_(0, indices, logits)

                if train_hypers['distributed']:
                    torch.distributed.barrier()
                    vote_logits = reduce_tensor(vote_logits, nprocs)
                    
                predict_labels = torch.argmax(vote_logits, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                val_pt_labs = raw_labels.cpu().detach().numpy()

                if train_hypers.local_rank == 0:
                    hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, unique_label))
                    pbar.update(1)

        if train_hypers.local_rank == 0:
            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str,iou):
                print('%s : %.2f%%' % (class_name, class_iou*100))
            val_miou = np.nanmean(iou) * 100
            pbar.close()
            print('Current val miou is %.3f ' % val_miou)
            print('Inference time per %d is %.4f seconds\n' %
                (dataset_config.val_data_loader.batch_size, np.mean(time_list)))

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    main(configs)
