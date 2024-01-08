from sample_factory.utils.utils import log


def interleave(*args):
    return [val for pair in zip(*args) for val in pair]


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def freeze_selected(cfg, model):
    if cfg.freeze_encoder:
        freeze(model.encoder)
        log.debug("Frozen encoder.")

    if cfg.freeze_core:
        freeze(model.core)
        freeze(model.decoder)
        log.debug("Frozen core.")

    if cfg.freeze_policy_head:
        freeze(model.action_parameterization)
        log.debug("Frozen policy head.")

    if cfg.freeze_critic_head:
        freeze(model.critic_linear)
        log.debug("Frozen critic head.")


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def unfreeze_selected(step, cfg, model, models_frozen):
    if step >= cfg.unfreeze_encoder and models_frozen["encoder"]:
        unfreeze(model.encoder)
        models_frozen["encoder"] = False
        log.debug("Unfrozen encoder.")

    if step >= cfg.unfreeze_core and models_frozen["core"]:
        unfreeze(model.core)
        unfreeze(model.decoder)
        models_frozen["core"] = False
        log.debug("Unfrozen core.")

    if step >= cfg.unfreeze_policy_head and models_frozen["policy_head"]:
        unfreeze(model.action_parameterization)
        models_frozen["policy_head"] = False
        log.debug("Unfrozen policy head.")

    if step >= cfg.unfreeze_critic_head and models_frozen["critic_head"]:
        unfreeze(model.critic_linear)
        models_frozen["critic_head"] = False
        log.debug("Unfrozen critic head.")
