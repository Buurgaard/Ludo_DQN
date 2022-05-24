import ludopy
import numpy as np
import torch
import math
import random
from collections import namedtuple, deque
from itertools import count

from torch import Tensor, nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os

from joblib import Parallel, delayed
import multiprocessing

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


# Define the Network Model
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(58 * 5 + 6, 64)
        self.bn1 = nn.BatchNorm1d(1)
        # self.layer2 = nn.Linear(512,256)
        # self.bn2 = nn.BatchNorm1d(1)
        #self.layer3 = nn.Linear(256, 64)
        #self.bn3 = nn.BatchNorm1d(1)
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(1)
        self.layer5 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.to(device)
        bn = x.shape[0]
        # print(x.shape, x[:,:,58*7:])
        peices = torch.reshape(x[:, :, :58 * 7], (bn, 7, 58))
        # print(peices)
        dice = x[:, :, -6:]
        out = torch.zeros((bn, 1, 4), device=device)
        end = peices[:, -3:]
        for i in range(4):
            conc = torch.cat((peices[:, :i], peices[:, i + 1:4]), 1)
            xs = torch.sum(conc, 1).unsqueeze(1)
            xi = peices[:, i].unsqueeze(1)

            input = torch.cat((torch.cat((xi, xs, end), 1).flatten(1).unsqueeze(1), dice), 2)
            # print(input)
            xi = F.relu(self.bn1(self.layer1(input)))
            # xi = F.relu(self.bn2(self.layer2(xi)))
            #xi = F.relu(self.bn3(self.layer3(xi)))
            xi = F.relu(self.bn4(self.layer4(xi)))
            xi = self.layer5(xi)
            # print(xi)
            # print(out[:,:,i])
            out[:, :, i] = xi.flatten(1)

        # print(out)
        return out


# Function to generate the input array
def build_input(dice, player_pieces, enemy_pieces):
    x = torch.zeros(58 * 7 + 6, 1)
    idx = 0
    for piece in player_pieces:
        x[58 * idx + piece] = 1
        idx += 1
    for enemy in enemy_pieces:
        for piece in enemy:
            x[58 * idx + piece] += 1

        idx += 1

    x[58 * idx + dice - 1] = 1

    return x.t()


# Function to generate the next state input array
def build_next_state(player_pieces, enemy_pieces):
    x = torch.zeros(58 * 7 + 6, 1)
    idx = 0
    for piece in player_pieces:
        x[58 * idx + piece] = 1
        idx += 1
    for enemy in enemy_pieces:
        for piece in enemy:
            x[58 * idx + piece] += 1

        idx += 1
    for dice in range(6):
        x[58 * idx + dice] = (1 / 6)

    return x.t()


# semi intelligent player ------------------------------------------------------------------
def piece_to_goal_lane(dice, player_pieces, move_pieces):
    pieces_to_goal_lane = []
    for piece in move_pieces:
        if player_pieces[piece] < 52 and player_pieces[piece] + dice > 51:
            pieces_to_goal_lane.append(piece)
    return pieces_to_goal_lane


def piece_to_goal(dice, player_pieces, move_pieces):
    pieces_to_goal = []
    for piece in move_pieces:
        if player_pieces[piece] + dice == 57:
            pieces_to_goal.append(piece)
    return pieces_to_goal


def out_of_home(dice, player_pieces, move_pieces):
    home_pieces = []
    if dice == 6:
        for piece in move_pieces:
            if player_pieces[piece] == 0:
                home_pieces.append(piece)
    return home_pieces


def blocked_tiles(enemy_pieces):
    tiles_blocked = []
    safe_tiles = [9, 14, 22, 27, 35, 40, 48]
    for enemy in enemy_pieces:
       # print('enemy:', enemy)
        for piece in enemy:
        #    print("piece: ", piece)
            if piece in safe_tiles or np.count_nonzero(enemy == piece) > 1:
                tiles_blocked.append(piece)
    return tiles_blocked


def move_furthest_piece(dice, player_pieces, move_pieces, enemy_pieces):
    #print("en2", enemy_pieces)
    tiles_blocked = blocked_tiles(enemy_pieces)
    furthest_piece = -1
    piece_to_move = -1
    for piece in move_pieces:
        if player_pieces[piece] + dice not in tiles_blocked:
            if player_pieces[piece] >= furthest_piece and player_pieces[piece] < 52:
                piece_to_move = piece
                furthest_piece = player_pieces[piece]
    return piece_to_move


def get_nearest(move_pieces, player_pieces):
    nearest_piece = 100
    piece_to_move = -1
    for piece in move_pieces:
        if player_pieces[piece] < nearest_piece:
            piece_to_move = piece
    return piece_to_move


def semi_int(dice, move_pieces, player_pieces, enemy_pieces):
    if len(move_pieces) == 1:
        return move_pieces[0]

    pieces_to_goal_lane = piece_to_goal_lane(dice, player_pieces, move_pieces)
    pieces_to_goal = piece_to_goal(dice, player_pieces, move_pieces)
    home_pieces = out_of_home(dice, player_pieces, move_pieces)
    fpiece_to_move = move_furthest_piece(dice, player_pieces, move_pieces, enemy_pieces)
    npiece_to_move = get_nearest(move_pieces, player_pieces)

    if pieces_to_goal_lane:
        return pieces_to_goal_lane[0]
    if pieces_to_goal:
        return pieces_to_goal[0]
    if home_pieces:
        return home_pieces[0]
    if fpiece_to_move != -1:
        return fpiece_to_move
    return npiece_to_move

#--------------------------------------------------------------------------------------------------------

# Hyperparameters
#BATCH_SIZE = 512
#GAMMA = 0.95
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 400
#TARGET_UPDATE = 1
NUM_EVAL = 600
#steps_done = 0

# init DQNs
policy_net = DQN().to(device)
# Load Checkpoint
checkpoint = torch.load("tests/2022-05-19_09-15-56/1/model_000800.state")
#checkpoint = torch.load("tests/2022-05-23_19-15-27/0/model_001496.state")
policy_net.load_state_dict(checkpoint)


# Select greedy action
def select_greedy_action(state, move_pieces):
    with torch.no_grad():
        policy_net.eval()
        output = policy_net(torch.unsqueeze(state, 0)).cpu()
        value_list = torch.index_select(output, 2, move_pieces)
        tmax = torch.max(value_list, 2)

        # print("tmax", tmax, tmax[1])
        return move_pieces[tmax[1].data[0]], tmax[0]


# Select random action
def select_random_action(move_pieces):
    # print("random")
    return move_pieces[np.random.randint(0, len(move_pieces))], None



# Evaluation
def evaluate(n_games):
    # Which players are random:

    def one_game():
        rand_players = [0,0,0,2]
        random.shuffle(rand_players)
        game = ludopy.Game()

        there_is_a_winner = False
        n_round = 0
        g3_goal = 0

        #print(rand_players)

        while not there_is_a_winner:
            n_round += 0.25
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = game.get_observation()

            action = -1

            if len(move_pieces) > 1:  # Actual decision

                if rand_players[player_i] == 1:  # if random
                    action, _ = select_random_action(move_pieces)

                elif rand_players[player_i] == 2:  # if semi
                    _, enemy_pieces_seen_from = game.get_pieces(seen_from=player_i)
                    for i in range(3):
                        for piece in range(4):
                            if enemy_pieces_seen_from[i][piece] != 0 and enemy_pieces_seen_from[i][piece] <= 51:
                                enemy_pieces_seen_from[i][piece] = (enemy_pieces_seen_from[i][piece]+(13 + i * 13)) % 52
                            else:
                                enemy_pieces_seen_from[i][piece] = 0
                    action = semi_int(dice, move_pieces, player_pieces, enemy_pieces_seen_from)

                else:  # AI
                    state = build_input(dice, player_pieces, enemy_pieces)
                    action, _ = select_greedy_action(state, torch.tensor(move_pieces))

            elif len(move_pieces) == 1:
                action = move_pieces[0]
            #print(player_i, rand_players[player_i])
            _, _, player_pieces_next, _, _, there_is_a_winner = game.answer_observation(action)

            n_in_goal = 0
            for pp in player_pieces_next:
                if pp == 57:
                    n_in_goal += 1
            if n_in_goal == 3 and g3_goal == 0:
                g3_goal = n_round

            if (there_is_a_winner):
                if g3_goal == 0:
                    g3_goal = n_round
                #print("Saving game video")
                #game.save_hist_video(f"game_video.mp4")
                return [rand_players[player_i] == 2, n_round, g3_goal]

    num_cores = multiprocessing.cpu_count()
    games = np.array(Parallel(n_jobs=num_cores)(delayed(one_game)() for i in range(n_games)))

    wins = sum(games[:, 0])
    rounds = sum(games[:, 1]) / n_games
    rounds_3 = sum(games[:, 2]) / n_games
    print("")
    print("Ai:", n_games - wins, (n_games - wins) / n_games)
    print("Ra:", wins, wins / n_games)
    print("Ro:", rounds)



    return [wins, n_games - wins, rounds, rounds_3]


folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir("tests/{}".format(folder))

num_episodes = 100

for i_episode in range(num_episodes):

    file = open("tests/{}/data.csv".format(folder), 'a')
    data = evaluate(NUM_EVAL)
    file.write("{:d},{:d},{:f},{:f},{:f},{:f}\n".format(i_episode, NUM_EVAL, data[0], data[1], data[2], data[3]))

    file.close()

    print("Epoch:", i_episode)

