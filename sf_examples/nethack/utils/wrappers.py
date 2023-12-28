"""Taken & adapted from Chaos Dwarf in Nethack Challenge Starter Kit:
https://github.com/Miffyli/nle-sample-factory-baseline


MIT License

Copyright (c) 2021 Anssi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import os
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import cv2
import gymnasium as gym
import imageio
import numpy as np
import render_utils
from ansi2image.ansi2image import Ansi2Image
from nle import nethack
from numba import njit
from PIL import Image, ImageDraw, ImageFont

from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sf_examples.nethack.utils.task_rewards import (
    EatingScore,
    GoldScore,
    ScoutScore,
    SokobanfillpitScore,
    SokobansolvedlevelsScore,
    StaircasePetScore,
    StaircaseScore,
)

SMALL_FONT_PATH = os.path.abspath("sf_examples/nethack/render_utils/Hack-Regular.ttf")

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


@njit
def _tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w,
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[:, h_pixel : h_pixel + char_height, w_pixel : w_pixel + char_width] = char_array[char, color]


def _initialize_char_array(font_size, rescale_font_size):
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
    dummy_text = "".join([(chr(i) if chr(i).isprintable() else " ") for i in range(256)])
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    char_width = rescale_font_size[0]
    char_height = rescale_font_size[1]

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
        blstats_cursor=False,
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.blstats_cursor = blstats_cursor

        self.half_crop_size = crop_size // 2
        self.output_height_chars = crop_size
        self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width,
        )

        obs_spaces = {"screen_image": gym.spaces.Box(low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8)}
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            if self.blstats_cursor:
                center_x, center_y = obs["blstats"][:2]
            else:
                center_y, center_x = obs["tty_cursor"]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )

        obs["screen_image"] = out_image
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self._render_text_to_image(obs)
        return obs, info


class RenderCharImagesWithNumpyWrapperV2(gym.Wrapper):
    """
    Same as V1, but simpler and faster.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)
        self.char_array = np.ascontiguousarray(self.char_array)
        self.crop_size = crop_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.chw_image_shape = (
            3,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        obs_spaces = {"screen_image": gym.spaces.Box(low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8)}
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                # if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _populate_obs(self, obs):
        screen = np.zeros(self.chw_image_shape, order="C", dtype=np.uint8)
        render_utils.render_crop(
            obs["tty_chars"],
            obs["tty_colors"],
            obs["tty_cursor"],
            self.char_array,
            screen,
            crop_size=self.crop_size,
        )
        obs["screen_image"] = screen

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._populate_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._populate_obs(obs)
        return obs, info


class SeedActionSpaceWrapper(gym.Wrapper):
    """
    To have reproducible decorrelate experience we need to seed action space
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.action_space.seed(seed=seed)
        return obs, info


class PrevActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = 0

        obs_spaces = {"prev_actions": self.env.action_space}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs):
        self.prev_action = 0
        obs, info = super().reset(**kwargs)
        obs["prev_actions"] = np.array([self.prev_action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.prev_action = action
        obs["prev_actions"] = np.array([self.prev_action])
        return obs, reward, terminated, truncated, info


class BlstatsInfoWrapper(gym.Wrapper):
    def step(self, action):
        # because we will see done=True at the first timestep of the new episode
        # to properly calculate blstats at the end of the episode we need to keep the last_observation around
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated | truncated:
            info["episode_extra_stats"] = self.add_more_stats(info, last_observation)

        return obs, reward, terminated, truncated, info

    def add_more_stats(self, info, last_observation):
        extra_stats = info.get("episode_extra_stats", {})
        blstats = BLStats(*last_observation[self.env.unwrapped._blstats_index])
        new_extra_stats = {
            "score": blstats.score,
            "turns": blstats.time,
            "dlvl": blstats.depth,
            "max_hitpoints": blstats.max_hitpoints,
            "max_energy": blstats.max_energy,
            "armor_class": blstats.armor_class,
            "experience_level": blstats.experience_level,
            "experience_points": blstats.experience_points,
        }
        return {**extra_stats, **new_extra_stats}


class TaskRewardsInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.tasks = [
            EatingScore(),
            GoldScore(),
            ScoutScore(),
            SokobanfillpitScore(),
            # SokobansolvedlevelsScore(), # TODO: it could have bugs, for now turn off
            StaircasePetScore(),
            StaircaseScore(),
        ]

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        for task in self.tasks:
            task.reset_score()

        return obs, info

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, terminated, truncated, info = super().step(action)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        end_status = info["end_status"]

        if terminated | truncated:
            info["episode_extra_stats"] = self.add_more_stats(info)

        # we will accumulate rewards for each step and log them when done signal appears
        for task in self.tasks:
            task.reward(self.env.unwrapped, last_observation, observation, end_status)

        return obs, reward, terminated, truncated, info

    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {task.name: task.score for task in self.tasks}
        return {**extra_stats, **new_extra_stats}


class RecordAnsi(gym.wrappers.RecordVideo):
    def __init__(
        self,
        env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 30,
    ):
        super().__init__(env, video_folder, episode_trigger, step_trigger, video_length, name_prefix)
        self.fps = fps

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
            episode_path = Path(self.video_recorder.path)
            self.ansi2video(episode_path)
        self.recording = False
        self.recorded_frames = 1

    def ansi2video(self, episode_path):
        with open(episode_path, "r") as file:
            outputs = json.load(file)

        o = Ansi2Image(width=800, height=600, font_name="JetBrains Mono Regular", font_size=8)

        video_path = episode_path.parent / (episode_path.stem + ".mp4")
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file_path = temp_file.name
            with imageio.get_writer(video_path, fps=self.fps) as writer:
                try:
                    for ansi_string in outputs["stdout"]:
                        o.loads(ansi_string[1])
                        o.calc_size(margin=0)
                        o.save_image(temp_file_path, format="png")

                        image = Image.open(temp_file_path)
                        writer.append_data(np.array(image))
                except Exception as e:
                    print(f"Error: {e}")

        # TODO: we would like to log videos to wandb from here, but we can't look at runner._upload_videos
