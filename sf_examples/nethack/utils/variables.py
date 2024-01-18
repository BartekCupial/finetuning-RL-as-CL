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


class Score(Variable):
    def __init__(self):
        super().__init__()
        self.monster_names = list(monster_data["Name"])

    def get_value(self, env, last_observation, observation, end_status):
        # According to wiki https://nethackwiki.com/wiki/Score `Killing a monster — worth 4 times the monster's experience points`,
        # troublesome so just detect if the monster is killed by us.
        # TODO: check if crushing the monster with the boulder or other edge cases also give up points

        blstats = BLStats(*observation[env.unwrapped._blstats_index])
        if last_observation == ():
            last_blstats = blstats
        else:
            last_blstats = BLStats(*last_observation[env.unwrapped._blstats_index])

        def parse_message(obs):
            char_array = [chr(i) for i in obs]
            message = "".join(char_array)
            # replace null bytes
            message = message.replace("\x00", "")
            return message

        def parse_tty(obs):
            tty_chars = obs[env._observation_keys.index("tty_chars")]
            tty_colors = obs[env._observation_keys.index("tty_colors")]
            tty_cursor = obs[env._observation_keys.index("tty_cursor")]
            print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))

        with open("logs.txt", "a+") as f:
            # negative score possible when we start a new game
            points = max(blstats.score - last_blstats.score, 0)
            if points != 0:
                message = parse_message(observation[env._message_index])
                last_message = parse_message(last_observation[env._message_index])

                # detect if monster was killed, some can be destroyed # TODO: handle ghost there are few different ghost names
                if (
                    re.match(
                        r".*You (kill|destroy) the (?:\w+ )?(?:{})!.*".format("|".join(self.monster_names)), message
                    )
                    or re.match(r".*You (kill|destroy) it!.*", message)
                    or re.match(r".*The (?:{}) is killed!.*".format("|".join(self.monster_names)), message)
                    or re.match(r".*is caught in the gas spore's explosion!.*", message)
                ):
                    # print(f"kill {match.group(1)}")
                    pass
                elif re.match(
                    r".*You hear the rumble of distant thunder.*", message
                ):  # you killed your pet, we should treat it differently
                    pass
                # detect if we gathered gold
                elif max(blstats.gold - last_blstats.gold, 0) > 0:
                    pass
                # detect if we identify the wand by engraving + 10 points
                elif re.match(r".*This ([\S]+) wand is a wand of ([\S]+).*", message) and points == 10:
                    pass
                # detect if we idenify the wand by engraving + 10 points
                elif re.match(r".*Do you want to add to the current engraving\?.*", message) and points == 10:
                    pass
                # detect if we identify the potion by drinking + 10 points
                elif re.match(r".*What do you want to drink\?.*", last_message) and points == 10:
                    pass
                elif last_blstats.depth - blstats.depth != 0:
                    # f.write(f"last_message: {last_message}, message: {message}, points: {points}\n")
                    # I think depending how many levels I drop I get more points
                    # 50 points for each level
                    # 1 -> 50
                    # 2 -> 100
                    # 3 -> 150
                    # 4+ -> 200+?
                    # we should skip this message and instead handle it in DungeonLevel Variable
                    pass
                else:
                    f.write(f"last_message: {last_message}, message: {message}, points: {points}\n")

        return self.value


class Monster(Variable):
    def get_value(self, env, last_observation, observation, end_status):
        # According to wiki https://nethackwiki.com/wiki/Score `Killing a monster — worth 4 times the monster's experience points`,
        # troublesome so just detect if the monster is killed by us.
        # TODO: check if crushing the monster with the boulder or other edge cases also give up points

        char_array = [chr(i) for i in observation[env._message_index]]
        message = "".join(char_array)

        # detect if monster was killed
        pattern = r"You kill the ([\w-]+(?: [\w-]+)?(?: [\w-]+)?)!"
        match = re.match(pattern, message)
        if match:
            # calculate the score for killing the monster
            blstats = BLStats(*observation[env.unwrapped._blstats_index])
            last_blstats = BLStats(*last_observation[env.unwrapped._blstats_index])
            score = blstats.score - last_blstats.score
            self.value += score

        return self.value


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


class DungeonLevel(Variable):
    def __init__(self):
        super().__init__()
        self.sokoban_level = None
        self.value = 1

    def reset_value(self):
        super().reset_value()
        self.sokoban_level = None
        self.value = 1

    def get_value(self, env, last_observation, observation, end_status):
        # According to wiki https://nethackwiki.com/wiki/Score we get
        # `50 * (DL-1) points where DL is the deepest level reached`

        blstats = BLStats(*observation[env.unwrapped._blstats_index])
        if last_observation == ():
            last_blstats = blstats
        else:
            last_blstats = BLStats(*last_observation[env.unwrapped._blstats_index])
        dungeon_level = blstats.depth

        # check if we are in sokoban
        # we have to calculate dlvl differently in sokoban https://nethackwiki.com/wiki/Dungeon_level
        if blstats.dungeon_number == 4:
            if self.sokoban_level is None:
                # save the level when sokoban first appeared
                self.sokoban_level = dungeon_level
            # set the difficulty to custom sokoban
            dungeon_level = 2 * self.sokoban_level - dungeon_level + 2

        if dungeon_level > self.value:
            self.value = dungeon_level

        return self.value


class DeepDungeonLevel(Variable):
    def __init__(self):
        super().__init__()
        self.deepest_level = 0

    def reset_value(self):
        self.deepest_level = 0
        return super().reset_value()

    def get_value(self, env, last_observation, observation, end_status):
        # According to wiki https://nethackwiki.com/wiki/Score we get
        # `1000 points for each dungeon level reached beyond 20, to a maximum of 10000 points`

        blstats = BLStats(*observation[env.unwrapped._blstats_index])
        dungeon_level = blstats.depth

        if dungeon_level > self.deepest_level:
            self.deepest_level = dungeon_level

            if dungeon_level > 20:
                self.value += 1
                self.value -= min(self.value - 10, 0)

        return self.value
