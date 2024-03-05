from pathlib import Path

import gym


class TtyrecInfoWrapper(gym.Wrapper):
    def step(self, action):
        ttyrec = self.env.unwrapped.gym_env.nethack._ttyrec
        obs, reward, done, info = self.env.step(action)

        if done:
            info["episode_extra_stats"]["ttyrecname"] = Path(ttyrec).name

        return obs, reward, done, info
