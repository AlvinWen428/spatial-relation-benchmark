import os
import argparse
from time import time
from datetime import datetime
from collections import defaultdict
from progressbar import ProgressBar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler

from models import build_model
from dataset import create_dataloader
import utils
from utils import TensorboardManager, RecordExp, flatten_dict, merge_dicts
from loss import BinaryCEWithLogitLoss
from optimizer import build_optimizer
from configs import get_cfg_defaults


DEVICE = ""


def get_inp(data_batch, device):
    predi = data_batch['predicate']['idx'].to(device, non_blocking=True)
    subj_bbox = data_batch['subject']['bbox'].to(device, non_blocking=True)
    obj_bbox = data_batch['object']['bbox'].to(device, non_blocking=True)
    rgb = data_batch['img'].to(device, non_blocking=True)

    inp = {
        "full_im": rgb,
        "bbox_s": subj_bbox,
        "bbox_o": obj_bbox,
        "predicate": predi
    }
    return inp


def validate(loader, model, criterion, device):
    """
    :param loader:
    :param model:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    correct = []
    tp = []
    fp = []
    p = []
    losses = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    with torch.no_grad():
        _bar = ProgressBar(max_value=len(loader))
        for i, data_batch in enumerate(loader):
            label = data_batch['label'].to(device, non_blocking=True)
            inp = get_inp(data_batch, device)

            output = model(**inp)
            loss = criterion(output, label.to(dtype=torch.float32), reduction='none')
            losses.append(loss)

            logit = output[0] if isinstance(output, tuple) else output
            batch_correct = (((logit > 0) & (label == True))
                             | ((logit <= 0) & (label == False))).tolist()
            tp.extend(((logit > 0) & (label == True)).tolist())
            fp.extend(((logit > 0) & (label == False)).tolist())

            correct.extend(batch_correct)
            p.extend((label == True).tolist())
            for pred_name, _correct in zip(data_batch['predicate']['name'],
                                           batch_correct):
                correct_rel[pred_name].append(_correct)
            _bar.update(i)

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())
    losses = torch.cat(losses, dim=0)

    return acc, pre, rec, f1, acc_rel, acc_rel_avg, losses


def train(loader, model, criterion, optimizer, device):
    model.train()
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    losses = []
    avg_loss = []
    correct = []
    tp = []
    fp = []
    p = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    time_last_batch_end = time()
    for i, data_batch in enumerate(loader):
        time_start = time()
        label = data_batch['label'].to(device, non_blocking=True)
        inp = get_inp(data_batch, device)
        output = model(**inp)

        loss = criterion(output, label.to(dtype=torch.float32))
        time_forward += (time() - time_start)

        logit = output[0] if isinstance(output, tuple) else output
        avg_loss.append(loss.item())
        losses.append(loss.item())
        batch_correct = (((logit > 0) & (label == True))
                         | ((logit <= 0) & (label == False))).tolist()
        correct.extend(batch_correct)
        tp.extend(((logit > 0) & (label == True)).tolist())
        p.extend((label == True).tolist())
        fp.extend(((logit > 0) & (label == False)).tolist())
        for pred_name, _correct in zip(data_batch['predicate']['name'],
                                       batch_correct):
            correct_rel[pred_name].append(_correct)

        optimizer.zero_grad()
        time_start = time()
        loss.backward()
        time_backward += (time() - time_start)
        optimizer.step()
        time_data_loading += (time_start - time_last_batch_end)
        time_last_batch_end = time()

        if i % 50 == 0:
            print(
                '[%d/%d] Loss = %.02f, Forward time = %.02f, Backward time = %.02f, Data loading time = %.02f' \
                % (i, len(loader), np.mean(avg_loss), time_forward,
                   time_backward, time_data_loading))

            avg_loss = []

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())
    losses = sum(losses) / len(losses)

    return acc, pre, rec, f1, acc_rel, acc_rel_avg, losses


def save_checkpoint(epoch, model, optimizer, acc, cfg):
    path = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}/model_best.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'accuracy': acc,
    }, path)
    print('Checkpoint saved to %s' % path)


def load_best_checkpoint(model, cfg, device):
    path = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)
    model.to(device)


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % model_path)
    model.to(device)


def entry_train(cfg, device, record_file=""):
    loader_train, _, _ = create_dataloader(split='train', **cfg.DATALOADER)
    loader_valid, _, _ = create_dataloader('valid', **cfg.DATALOADER)
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)

    model = build_model(cfg)
    model.to(device)

    if utils.is_dist_avail_and_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(cfg, model)
    if cfg.TRAIN.scheduler == 'plateau':
        scheduler = PlateauLRScheduler(optimizer, mode='max', decay_rate=cfg.TRAIN.lr_decay_ratio, patience_t=cfg.TRAIN.patience, warmup_t=cfg.TRAIN.warmup, verbose=True)
    elif cfg.TRAIN.scheduler == 'cosine':
        scheduler = CosineLRScheduler(optimizer, t_initial=cfg.TRAIN.num_epochs, warmup_t=cfg.TRAIN.warmup, lr_min=0.0)
    else:
        raise ValueError

    if utils.is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[utils.get_rank()], output_device=utils.get_rank(), find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    print(model)

    criterion = BinaryCEWithLogitLoss()

    best_acc_rel_avg_valid = -1
    best_epoch_rel_avg_valid = 0
    best_acc_rel_avg_test = -1
    early_stop_flag = torch.zeros(1).to(device)

    log_dir = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}"
    os.makedirs(log_dir, exist_ok=True)
    tb = TensorboardManager(log_dir)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        cfg.dump(stream=f)
    for epoch in range(cfg.TRAIN.num_epochs):
        print('\nEpoch #%d' % epoch)
        if hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)

        print('Training..')
        (acc_train, pre_train, rec_train, f1_train, acc_rel_train,
         acc_rel_avg_train, loss_train) = train(loader_train, model, criterion, optimizer, device)
        print(f'Train, acc avg: {acc_rel_avg_train} acc: {acc_train},'
              f' pre: {pre_train}, rec: {rec_train}, f1: {f1_train}, loss: {loss_train}, lr: {optimizer.param_groups[-1]["lr"]}')
        print({x: round(y, 3) for x, y in acc_rel_train.items()})
        tb.update('train', epoch, {'acc': acc_train, 'lr': optimizer.param_groups[-1]['lr']})

        acc_valid_tensor = torch.zeros(1).to(device)
        if utils.get_rank() == 0:
            print('\nValidating..')
            (acc_valid, pre_valid, rec_valid, f1_valid, acc_rel_valid,
             acc_rel_avg_valid, loss_valid) = validate(loader_valid, model_without_ddp, criterion, device)
            print(f'Valid, acc avg: {acc_rel_avg_valid} acc: {acc_valid},'
                  f' pre: {pre_valid}, rec: {rec_valid}, f1: {f1_valid}, loss: {loss_valid.mean().item()}')
            print({x: round(y, 3) for x, y in acc_rel_valid.items()})
            tb.update('val', epoch, {'acc': acc_valid})
            acc_valid_tensor = torch.tensor([acc_valid]).to(device)

            print('\nTesting..')
            (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
             acc_rel_avg_test, loss_test) = validate(loader_test, model_without_ddp, criterion, device)
            print(f'Test, acc avg: {acc_rel_avg_test} acc: {acc_test},'
                  f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss: {loss_test.mean().item()}')
            print({x: round(y, 3) for x, y in acc_rel_test.items()})

            if acc_rel_avg_valid > best_acc_rel_avg_valid:
                print('Accuracy has improved')
                best_acc_rel_avg_valid = acc_rel_avg_valid
                best_epoch_rel_avg_valid = epoch

                save_checkpoint(epoch, model_without_ddp, optimizer, acc_rel_avg_valid, cfg)
            if acc_rel_avg_test > best_acc_rel_avg_test:
                best_acc_rel_avg_test = acc_rel_avg_test

            if (epoch - best_epoch_rel_avg_valid) > cfg.TRAIN.early_stop:
                early_stop_flag += 1

        utils.synchronize()
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(acc_valid_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(early_stop_flag, op=torch.distributed.ReduceOp.SUM)
            if early_stop_flag >= 1:
                print(f"Early stopping at {epoch} as val acc did not improve"
                      f" for {cfg.TRAIN.early_stop} epochs.")
                break
        else:
            if early_stop_flag == 1:
                print(f"Early stopping at {epoch} as val acc did not improve"
                      f" for {cfg.TRAIN.early_stop} epochs.")
                break

        scheduler.step(epoch=epoch+1, metric=acc_valid_tensor)

    if utils.get_rank() == 0:
        print('\nTesting..')
        load_best_checkpoint(model_without_ddp, cfg, device)
        (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
         acc_rel_avg_test, loss_test) = validate(loader_test, model_without_ddp, criterion, device)
        print(f'Best valid, acc: {best_acc_rel_avg_valid}')
        print(f'Best test, acc: {best_acc_rel_avg_test}')
        print(f'Test at best valid, acc avg: {acc_rel_avg_test}, acc: {acc_test},'
              f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss_test: {loss_test.mean().item()}')
        print({x: round(y, 3) for x, y in acc_rel_test.items()})

        if record_file != "":
            exp = RecordExp(record_file)
            exp.record_param(flatten_dict(dict(cfg)))
            exp.record_result({
                "final_train": acc_rel_avg_train,
                "best_val": best_acc_rel_avg_valid,
                "best_test": best_acc_rel_avg_test,
                "final_test": acc_rel_avg_test
            })


def entry_test(cfg, model_path, device):
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)
    criterion = BinaryCEWithLogitLoss()
    model = build_model(cfg)
    model.to(device)
    load_checkpoint(model, model_path, device)

    print('\nTesting..')
    (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
     acc_rel_avg_test, loss_test) = validate(loader_test, model, criterion, device)
    print(f'Test at best valid, acc avg: {acc_rel_avg_test}, acc: {acc_test}, '
          f'pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss: {loss_test.mean().item()}')
    print({x: round(y, 3) for x, y in acc_rel_test.items()})
    for x, y in acc_rel_test.items():
        print(x, ':', round(y, 3))
    return {'acc avg': acc_rel_avg_test, 'acc': acc_test, 'precision': pre_test, 'recall': rec_test, 'f1': f1_test,
            'acc of class': acc_rel_test, 'loss': loss_test.mean().item()}


def entry_batch_test(model_paths, device):
    each_class_acc_dicts = []
    result_dicts = []
    for path in model_paths:
        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(os.path.join(path, 'config.yaml'))
        _cfg.merge_from_list(cmd_args.opts)
        _cfg.freeze()

        utils.set_seed(_cfg.EXP.SEED)

        model_path = os.path.join(path, 'model_best.pth')
        test_results = entry_test(_cfg, model_path, device)
        each_class_acc_dicts.append(test_results.pop('acc of class'))
        result_dicts.append(test_results)

    each_class_accuracies = merge_dicts(each_class_acc_dicts)
    final_result = merge_dicts(result_dicts)

    print()
    print("\U0001F680\U0001F680\U0001F680 Average Results over all models: \U0001F680\U0001F680\U0001F680")
    for predicate in each_class_accuracies:
        print('{} : {:.4f}\pm{:.4f}'.format(predicate, np.mean(each_class_accuracies[predicate]).item(), np.std(each_class_accuracies[predicate]).item()))

    print("\U0001F973 Accuracy of each predicate (in Latex format):")
    predicate_names = list(each_class_accuracies.keys())
    sorted(predicate_names)
    predicate_names = [predicate_names[i:i+5] for i in range(0, len(predicate_names), 5)]
    for name_list in predicate_names:
        print(" & ".join(name_list))
        for name in name_list:
            print("${:.2f}\pm{:.2f}$ & ".format(100 * np.mean(each_class_accuracies[name]).item(),
                                                100 * np.std(each_class_accuracies[name]).item()), end='')
        print()

    print("\U0001F60E Overall Results (in Latex format):")
    metrics = ['acc avg', 'acc', 'precision', 'recall', 'f1', 'loss']
    for m in metrics:
        print('${:.4f}\pm{:.4f}$ & '.format(np.mean(final_result[m]).item(), np.std(final_result[m]).item()), end=' ')
    print()


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--model-path', type=str, nargs='+', default="")
    parser.add_argument('--record-file', type=str, default="")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.merge_from_list(cmd_args.opts)
        if _cfg.EXP.EXP_ID == "":
            _cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        _cfg.freeze()

        utils.init_distributed_mode(cmd_args.local_rank)
        utils.set_seed(_cfg.EXP.SEED)
        DEVICE = torch.device("cuda", utils.get_rank())

        print(_cfg)

        entry_train(_cfg, DEVICE, cmd_args.record_file)

    elif cmd_args.entry == "test":
        assert not cmd_args.exp_config == ""
        assert len(cmd_args.model_path) == 1, "Only one model path is allowed for test flag"

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.merge_from_list(cmd_args.opts)
        _cfg.freeze()
        print(_cfg)

        utils.set_seed(_cfg.EXP.SEED)
        entry_test(_cfg, cmd_args.model_path, DEVICE)

    elif cmd_args.entry == "batch-test":
        assert len(cmd_args.model_path) > 1, "At least one model path is required for batch-test flag"
        entry_batch_test(cmd_args.model_path, DEVICE)

    else:
        raise ValueError(f"Invalid entry: {cmd_args.entry}")
