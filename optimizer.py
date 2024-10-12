import json
import torch


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name[16:].split('.')[0])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def build_optimizer(cfg, model):
    if cfg.TRAIN.layer_decay < 1.0:
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "blocks"):
                num_layers = len(model.backbone.blocks)
            else:
                assert hasattr(model, "blocks")
                num_layers = len(model.blocks)
        else:
            assert hasattr(model, "blocks")
            num_layers = len(model.blocks)
        assigner = LayerDecayValueAssigner(
            list(cfg.TRAIN.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    param_groups = get_parameter_groups(model, weight_decay=cfg.TRAIN.l2,
                                        get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                        get_layer_scale=assigner.get_scale if assigner is not None else None,
                                        skip_list=model.no_weight_decay() if hasattr(model, 'no_weight_decay') else ())

    if cfg.TRAIN.optimizer == 'adam':
        optim = torch.optim.Adam(param_groups, lr=cfg.TRAIN.learning_rate)
    elif cfg.TRAIN.optimizer == 'adamw':
        optim = torch.optim.AdamW(param_groups, lr=cfg.TRAIN.learning_rate)
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optim = torch.optim.RMSprop(param_groups, lr=cfg.TRAIN.learning_rate)
    elif cfg.TRAIN.optimizer == 'sgd':
        optim = torch.optim.SGD(param_groups, lr=cfg.TRAIN.learning_rate, momentum=0.9)
    else:
        raise ValueError

    return optim



