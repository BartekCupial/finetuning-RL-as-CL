import re
from enum import Enum, auto

import numpy as np
import pandas as pd
from nle import nethack

from sf_examples.nethack.utils.blstats import BLStats

monster_data = pd.read_csv("sf_examples/nethack/utils/reward_shaping/monster_data.csv")


class Variable:
    def __init__(self):
        self.value = 0
        # convert name to snake_case
        # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        self.name = re.sub("(?!^)([A-Z]+)", r"_\1", self.__class__.__name__).lower()

    def reset_value(self):
        self.value = 0


class Resistance(Variable):
    class ResistanceType(Enum):
        ACID = auto()
        COLD = auto()
        DISINTEGRATION = auto()
        FIRE = auto()
        MAGIC = auto()
        POISON = auto()
        SHOCK = auto()
        SLEEP = auto()
        HALLUCINATION = auto()
        DRAIN = auto()
        STONING = auto()
        SLIMING = auto()
        SICKNESS = auto()

    def get_value(self, env, last_observation, observation, end_status):
        # TODO: test if we've gained resistance
        # You feel full of hot air. | You feel warm! -> means that we've gained cold resistance
        pass
