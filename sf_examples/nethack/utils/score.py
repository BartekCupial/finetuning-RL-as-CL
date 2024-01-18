import re

import numpy as np
import pandas as pd
from nle import nethack

from sf_examples.nethack.utils.blstats import BLStats

monster_data = pd.read_csv("sf_examples/nethack/utils/reward_shaping/monster_data.csv")


class Score:
    def __init__(self):
        self.monster_names = list(monster_data["Name"])
        self.score_functions = [
            self.kill_monster,
            self.kill_pet,
            self.identify_wand,
            self.identify_potion,
            self.identify_scroll,
            self.eat_tripe_ration,
            self.quaffle_from_sink,
            self.read_novel,
            self.reach_oracle,
            self.gold,
            self.level_reached,
            self.deep_level_reached,
        ]
        self.score_functions_keys = {f.__name__.upper(): f.__name__.upper() for f in self.score_functions}
        self.score_functions_keys["identify_wand".upper()] = "IDENTIFY"
        self.score_functions_keys["identify_potion".upper()] = "IDENTIFY"
        self.score_functions_keys["identify_scroll".upper()] = "IDENTIFY"

        self.scores = {v: 0 for v in self.score_functions_keys.values()}

        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def reset(self):
        self.scores = {v: 0 for v in self.score_functions_keys.values()}

    def kill_monster(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Killing a monster — worth 4 times the monster's experience points

        # there may be more ways to kill a monster, not sure how to know them
        # TODO: check if crushing the monster with the boulder or other edge cases also give up points # The poison was deadly...
        return (
            re.match(r".*You (kill|destroy) the (?:\w+ )?(?:{})!.*".format("|".join(self.monster_names)), message)
            or re.match(r".*You (kill|destroy) it!.*", message)
            or re.match(r".*The (?:{}) is killed!.*".format("|".join(self.monster_names)), message)
            or re.match(r".*caught in the gas spore's explosion!.*", message)
            or re.match(r".*You kill .* of Aigoruun!.*", message)  # TODO: add him to list of the monsters
            or re.match(r".*The poison was deadly\.\.\..*", message)
            or re.match(
                r".*You (kill|destroy) the .*!.*", message
            )  # TODO: handle halluc monsters, right now there is a possibility that we destroyed an item.
        )

    def kill_pet(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # I figured it would be useful to treat pet differently
        return re.match(r".*You hear the rumble of distant thunder.*", message)

    def identify_wand(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Identifying a wand by engraving or zapping — 10 points
        #   Non-directional wands always give score
        #   Ray-type wands give score if zapped in any direction except yourself, or if engraved with
        #   Beam/immediate-type wands give score if zapped up or down, but not in other directions or at a monster engulfing you
        #   Breaking wands never gives score
        return (
            re.match(r".*This ([\S]+) wand is a wand of ([\S]+).*", message)
            or re.match(r".*Do you want to add to the current engraving\?.*", message)
            or re.match(r".*What do you want to zap\?.*", last_message)
        ) and points == 10

    def identify_potion(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Identifying a potion by quaffing — 10 points
        #   Lighting a potion of oil identifies it without giving points
        return re.match(r".*What do you want to drink\?.*", last_message) and points == 10

    def identify_scroll(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Identifying a scroll by reading — 10 points
        return re.match(r".*What do you want to read\?.*", last_message) and points == 10

    def identify_unknown(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # we don't know what is it, but we assume it's identification score
        return points == 10

    def oil_squeaky_board_trap(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Oiling or greasing a squeaky board trap — 9 points
        return False

    def eat_tripe_ration(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Eating a tripe ration when not an orc or carnivorous non-humanoid — 4 points
        return (
            re.match(r".*You finish eating the tripe ration\..*", message)
            or re.match(r".*You feel guilty.  Yak - dog food!.*", message)
        ) and points == 4

    def quaffle_from_sink(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Quaffing from a sink and getting the message "Yuk, this water tastes awful" — 4 points
        return re.match(r".*Yuk, this water tastes awful.*", last_message) and points == 4

    def read_novel(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # Reading your first novel — 80 points
        # TODO: handle this, for now this isn't neccessary since it gives the models only 80 points only once
        return False

    def reach_oracle(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # The first oracle of each type:
        #   Minor (if not yet major): 21 points
        #   Minor (if already major): 9 points
        #   Major (if not yet minor): 210 + (21 * XL) points
        #   Major (if already minor): 90 + (9 * XL) points
        XL = blstats.experience_level
        target_points = [9, 21, 210 + (21 * XL), 90 + (9 * XL)]

        x, y = blstats.x, blstats.y
        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]

        return np.any(neighbors == self.oracle_glyph) and points in target_points

    def gold(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # At the end of the game (via death, ascension or escape):
        #   1 point for each zorkmid more than starting gold (* 90% if game ended in death)
        return max(blstats.gold - last_blstats.gold, 0) == points

    def level_reached(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # At the end of the game (via death, ascension or escape):
        #   50 * (DL-1) points where DL is the deepest level reached
        return (blstats.depth - last_blstats.depth != 0) and points % 50 == 0

    def deep_level_reached(self, last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
        # At the end of the game (via death, ascension or escape):
        #   1000 points for each dungeon level reached beyond 20, to a maximum of 10000 points
        return (blstats.depth > 20) and points % 1000 == 0

    def get_value(self, env, last_observation, observation, end_status):
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
            # useful for debugging
            tty_chars = obs[env._observation_keys.index("tty_chars")]
            tty_colors = obs[env._observation_keys.index("tty_colors")]
            tty_cursor = obs[env._observation_keys.index("tty_cursor")]
            print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))

        # negative score possible when we start a new game
        points = max(blstats.score - last_blstats.score, 0)
        if points != 0:
            message = parse_message(observation[env._message_index])
            last_message = parse_message(last_observation[env._message_index])

            glyphs = observation[env._glyph_index]
            last_glyphs = last_observation[env._glyph_index]

            some = False
            for score_fn in self.score_functions:
                if score_fn(last_blstats, blstats, last_glyphs, glyphs, last_message, message, points):
                    some = True
                    key = self.score_functions_keys[score_fn.__name__.upper()]
                    self.scores[key] += points

            if not some:
                if self.identify_unknown(last_blstats, blstats, last_message, message, points):
                    key = "IDENTIFY"
                    self.scores[key] += points

                # with open("logs.txt", "a+") as f:
                #     f.write(f"last_message: {last_message}, message: {message}, points: {points}\n")

        return self.scores
