from typing import Optional

from nle.env.tasks import NetHackScore

from sample_factory.algo.utils.gymnasium_utils import patch_non_gymnasium_env
from sample_factory.utils.utils import videos_dir
from sf_examples.nethack.datasets.env import NetHackTtyrec
from sf_examples.nethack.utils.tasks import (
    NetHackChallenge,
    NetHackEat,
    NetHackGold,
    NetHackOracle,
    NetHackScout,
    NetHackStaircase,
    NetHackStaircasePet,
)
from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    PrevActionWrapper,
    RecordAnsi,
    RenderCharImagesWithNumpyWrapperV2,
    RenderWrapper,
    SeedActionSpaceWrapper,
    TaskRewardsInfoWrapper,
)

NETHACK_ENVS = dict(
    staircase=NetHackStaircase,
    score=NetHackScore,
    pet=NetHackStaircasePet,
    oracle=NetHackOracle,
    gold=NetHackGold,
    eat=NetHackEat,
    scout=NetHackScout,
    challenge=NetHackChallenge,
    dataset=NetHackTtyrec,
)


def nethack_env_by_name(name):
    if name in NETHACK_ENVS.keys():
        return NETHACK_ENVS[name]
    else:
        raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    env_class = nethack_env_by_name(env_name)

    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        # "specials",
        # "colors",
        # "chars",
        # "glyphs",
        # "inv_glyphs",
        # "inv_strs",
        # "inv_letters",
        # "inv_oclasses",
    )

    kwargs = dict(
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        gameloaddir=cfg.gameloaddir,
    )
    if env_name == "challenge":
        kwargs["no_progress_timeout"] = 150

    if env_name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")
    if cfg.state_counter is not None:
        kwargs.update(state_counter=cfg.state_counter)

    if env_name == "dataset":
        env = env_class(cfg, **kwargs)
    else:
        env = env_class(**kwargs)

    env = patch_non_gymnasium_env(env)

    if cfg.serial_mode and cfg.num_workers == 1:
        # full reproducability can only be achieved in serial mode and when there is only 1 worker
        env = SeedActionSpaceWrapper(env)

    if cfg.add_image_observation:
        env = RenderCharImagesWithNumpyWrapperV2(
            env,
            crop_size=cfg.crop_dim,
            rescale_font_size=(cfg.pixel_size, cfg.pixel_size),
        )

    if cfg.use_prev_action:
        env = PrevActionWrapper(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)

    if render_mode:
        env = RenderWrapper(env, render_mode)

    if env_config:
        if env_config["vector_index"] == 0 and env_config["worker_index"] == 0:
            if cfg.capture_video:
                env = RecordAnsi(env, videos_dir(cfg=cfg), episode_trigger=lambda t: t % cfg.capture_video_ith == 0)

    return env
