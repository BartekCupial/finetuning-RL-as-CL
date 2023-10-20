import sys
from os.path import join

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.context import global_model_factory, sf_global_context
from sample_factory.cfg.arguments import load_from_path, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import ActorCritic, default_make_actor_critic_func
from sample_factory.model.encoder import Encoder
from sample_factory.train import run_rl
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log
from sf_examples.nethack.algo.learning.learner import DatasetLearner
from sf_examples.nethack.models.chaotic_dwarf import ChaoticDwarvenGPT5
from sf_examples.nethack.models.kickstarter import KickStarter
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_learner,
    add_extra_params_model,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)
from sf_examples.nethack.nethack_utils import NETHACK_ENVS, make_nethack_env


def register_nethack_envs():
    for env_name in NETHACK_ENVS.keys():
        register_env(env_name, make_nethack_env)


def make_nethack_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    # TODO: register more possible models, here we should use parameters for specifying which model we want to use
    return ChaoticDwarvenGPT5(cfg, obs_space)


def load_pretrained_checkpoint(model, checkpoint_dir, checkpoint_kind):
    name_prefix = dict(latest="checkpoint", best="best")[checkpoint_kind]
    checkpoints = Learner.get_checkpoints(join(checkpoint_dir, "checkpoint_p0"), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, "cpu")
    model.load_state_dict(checkpoint_dict["model"])


def make_nethack_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    create_model = default_make_actor_critic_func

    use_distillation_loss = cfg.distillation_loss_coeff > 0.0
    use_kickstarting_loss = cfg.kickstarting_loss_coeff > 0.0

    if use_distillation_loss or use_kickstarting_loss:
        student = create_model(cfg, obs_space, action_space)
        if cfg.use_pretrained_checkpoint:
            load_pretrained_checkpoint(student, cfg.model_path, cfg.load_checkpoint_kind)
            log.debug("Loading student from pretrained checkpoint")

        teacher_cfg = load_from_path(join(cfg.teacher_path, "config.json"))
        teacher = create_model(teacher_cfg, obs_space, action_space)
        load_pretrained_checkpoint(teacher, cfg.teacher_path, cfg.load_checkpoint_kind)

        model = KickStarter(student, teacher, run_teacher_hs=cfg.run_teacher_hs)
        log.debug("Created kickstarter")
    else:
        model = create_model(cfg, obs_space, action_space)
        if cfg.use_pretrained_checkpoint:
            load_pretrained_checkpoint(model, cfg.model_path, cfg.load_checkpoint_kind)
            log.debug("Loading model from pretrained checkpoint")

    return model


def register_nethack_learner():
    sf_global_context().learner_cls = DatasetLearner


def register_nethack_components():
    register_nethack_envs()
    register_nethack_learner()
    global_model_factory().register_encoder_factory(make_nethack_encoder)
    global_model_factory().register_actor_critic_factory(make_nethack_actor_critic)


def parse_nethack_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_learner(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()
    cfg = parse_nethack_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
