import numpy as np


def calc_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    prev_owned = prev_obs['ball_owned_team']
    owned = obs['ball_owned_team']

    ball_position_r = 0.0

    player_prev = prev_obs['ball_owned_player']
    player_curr = obs['ball_owned_player']

    if owned == 0:
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = 1.0
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = 2.0
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = 2.5
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = 5.0
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = 3.0
        else:
            ball_position_r = 0.0

    elif owned == 1:
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = -1.0
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = -2.0
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = -2.5
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            ball_position_r = -5.0
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            ball_position_r = -3.0
        else:
            ball_position_r = 0.0

    else:
        ball_position_r = 0.0

    own_reward = 0.0
    if prev_owned == 1 and owned == 0:
        own_reward = 1.0
    elif prev_owned == 0 and owned == 1:
        own_reward = -1.0

    # left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
    # right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
    # yellow_r = right_yellow - left_yellow

    pass_r = 0.0
    if player_prev != player_curr and prev_owned == owned:
        pass_r = 1.0
    elif player_prev != player_curr and prev_owned != owned:
        pass_r = -1.0

    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0

    reward = 10.0 * win_reward + 10.0 * rew + 0.0001 * ball_position_r + 0.1 * own_reward + 0.1 * pass_r

    return reward
