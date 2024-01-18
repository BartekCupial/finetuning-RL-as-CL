import pandas as pd

monster_data = pd.read_csv("sf_examples/nethack/utils/reward_shaping/monster_data.csv")


def monster_score(monster_name, dungeon_level, player_level):
    # inspied by https://www.steelypips.org/nethack/experience-spoiler.html

    monster_exp = (monster_data[monster_data["Name"] == monster_name]["Experience"]).iloc[0]
    monster_level = (monster_data[monster_data["Name"] == monster_name]["Level"]).iloc[0]

    # Adjust for dungeon depth
    if monster_level > dungeon_level:
        monster_level -= 1
    elif monster_level < dungeon_level:
        monster_level += (dungeon_level - monster_level) // 5

    # Adjust for player level
    if monster_level < player_level:
        monster_level += monster_level // 4

    # Final XP calculation
    final_xp = monster_exp + (2 * monster_level + 1)

    score = 4 * final_xp

    return score


# # Example usage:
# monster_name = "fox"
# current_dungeon_level = 4
# current_player_level = 2

# resulting_score = monster_score(monster_name, current_dungeon_level, current_player_level)
# print(f"Score gained for killing the monster: {resulting_score}")
