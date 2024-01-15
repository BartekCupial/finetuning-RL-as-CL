import json
import tempfile
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio
import numpy as np
from ansi2image.ansi2image import Ansi2Image
from PIL import Image


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
