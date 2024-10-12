from collections import defaultdict, OrderedDict, MutableMapping, Hashable
import os
import random
import math
import csv
import numpy as np
import torch
import torch.distributed as dist
import tensorboardX


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(local_rank):
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        print('| distributed init (rank {})'.format(local_rank), flush=True)
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        setup_for_distributed(local_rank == 0)
    else:
        print('Not using distributed mode')
        return


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def set_seed(seed):
    if is_dist_avail_and_initialized():
        torch.manual_seed(seed + get_rank())
        np.random.seed(seed + get_rank())
        random.seed(seed + get_rank())
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def merge_dicts(dict_list):
    output_dict = {}
    for single_dict in dict_list:
        for k, v in single_dict.items():
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(v)
    return output_dict


class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        if get_rank() == 0:
            for k, v in vals.items():
                self.writer.add_scalar('%s_%s' % (split, k), v, step)


class RecordExp:
    def __init__(self, file_name):
        self.file_name = file_name
        self.param_recorded = False
        self.result_recorded = False

    def record_param(self, param_dict):
        """
        all parameters must be given at the same time. parameters must be given
        before the results
        :return:
        """
        assert not self.param_recorded
        self.param_recorded = True
        self.param_dict = param_dict

    def record_result(self, result_dict):
        """
        all results must be given at the same time
        :return:
        """
        assert self.param_recorded
        assert not self.result_recorded
        self.result_recorded = True

        if os.path.exists(self.file_name):
            with open(self.file_name, 'r') as csv_file:
                reader = csv.reader(csv_file)
                fields = next(reader)
        else:
            print("This is the first record of the experiment")
            fields = list(self.param_dict.keys()) + list(result_dict.keys())
            with open(self.file_name, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(fields)

        self.param_dict.update(result_dict)

        values = []
        for field in fields:
            if field in self.param_dict:
                values.append(self.param_dict[field])
            else:
                values.append("NOT PRESENT")

        extra_fields = list(set(self.param_dict.keys() - set(fields)))
        if not len(extra_fields) == 0:
            for field in extra_fields:
                values.append(f"{field:} {self.param_dict[field]}")
                print(f"adding extra field {field}")

        with open(self.file_name, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(values)


# source: https://stackoverflow.com/questions/2363731/append-new-row-to-old-csv-file-python
def flatten_dict(d, parent_key='', sep='_', use_short_name=True):
    items = []
    for k, v in d.items():
        if use_short_name:
            k, v = short_name(k), short_name(v)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


SHORT_NAME = {
    'DATALOADER': 'DL',
    'batch_size': 'bs',
    'datapath': 'dp',
    'load_image': 'lm',
    'crop': 'cr',
    'norm_data': 'nd',
    'data_aug_shift': 'das',
    'data_aug_color': 'dac',
    'resize_mask': 'rm',
    'TRAIN' : 'TR',
    'num_epochs': 'ne',
    'learning_rate': 'lr',
    'MODEL': 'M',
    'TWO_D': '2D',
    'feature_dim': 'fd',
    'LANGUAGE': 'LG',
    'DRNET': 'DR',
    'pretrained': 'pr',
    'dropout': 'dr',
    'num_layers': 'nl',
    'backbone': 'bb',
    'two_stream': '2s',
    'only_2d': 'o2d',
    'only_appr': 'oa',
    'VTRANSE': 'VT',
    'visual_feature_size': 'vfs',
    'predicate_embedding_dim': 'ped',
    'feat_size': 'fs',
    'feat_dim': 'fd',
    'roi_size': 'roi',
    'with_rgb': 'rgb',
    'with_depth': 'depth',
    'with_bbox': 'bbox',
    'add_union_feat': 'auf',
    '20200207_c_0.9_c_0.1_c_1.0.json': '20200207_def',
    '20200215_c_0.9_c_0.1_c_1.0.json': '20200215_def',
    '20200220_c_0.9_c_0.1_c_1.0.json': '20200220_def',
    'True' : 'T',
    'False': 'F',
    'trans_vec': 'tv',
    'raw_absolute': 'ra',
    'aligned_absolute': 'aa',
    'aligned_relative': 'ar',
    "with_class": 'wc',
    True: 'T',
    False: 'F',
    'remove_near_far': 'NO_N_F'
}


def short_name(x):
    if isinstance(x, Hashable):
        if x in SHORT_NAME:
            return SHORT_NAME[x]
        else:
            return x
    else:
        return x
