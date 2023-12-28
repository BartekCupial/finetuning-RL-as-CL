import numpy as np
from nle.nethack.actions import ACTIONS

ACTION_MAPPING = np.zeros(256)
for i, a in enumerate(ACTIONS):
    ACTION_MAPPING[a.value] = i
