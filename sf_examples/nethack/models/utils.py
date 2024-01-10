import sys

import torch.nn as nn

from sample_factory.utils.utils import log


def interleave(*args):
    return [val for pair in zip(*args) for val in pair]


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.trainable = False
            module.track_running_stats = False


def freeze_selected(step, cfg, model, models_frozen):
    for module_name, module_freeze in cfg.freeze.items():
        module_unfreeze = cfg.unfreeze.get(module_name, sys.maxsize)
        if step >= module_freeze and step <= module_unfreeze and not models_frozen[module_name]:
            freeze(getattr(model, module_name))
            log.debug(f"Frozen {module_name}.")
            models_frozen[module_name] = True

            if cfg.freeze_batch_norm:
                freeze_batch_norm(getattr(model, module_name))


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def unfreeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.trainable = True
            module.track_running_stats = True


def unfreeze_selected(step, cfg, model, models_frozen):
    for module_name, module_unfreeze in cfg.unfreeze.items():
        if step >= module_unfreeze and models_frozen[module_name]:
            unfreeze(getattr(model, module_name))
            log.debug(f"Unfrozen {module_name}.")
            models_frozen[module_name] = False

            if cfg.freeze_batch_norm:
                unfreeze_batch_norm(getattr(model, module_name))
