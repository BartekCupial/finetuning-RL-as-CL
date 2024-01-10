import sys

from sample_factory.utils.utils import log


def interleave(*args):
    return [val for pair in zip(*args) for val in pair]


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def freeze_selected(step, cfg, model, models_frozen):
    for module_name, module_freeze in cfg.freeze.items():
        module_unfreeze = cfg.unfreeze.get(module_name, sys.maxsize)
        if step >= module_freeze and step <= module_unfreeze and not models_frozen[module_name]:
            freeze(getattr(model, module_name))
            log.debug(f"Frozen {module_name}.")
            models_frozen[module_name] = True


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def unfreeze_selected(step, cfg, model, models_frozen):
    for module_name, module_unfreeze in cfg.unfreeze.items():
        if step >= module_unfreeze and models_frozen[module_name]:
            freeze(getattr(model, module_name))
            log.debug(f"Unfrozen {module_name}.")
            models_frozen[module_name] = False
