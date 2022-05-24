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

# Define replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_q_values', 'reward'))


class ReplayMemory(object):
    # Init the memory
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # Save a Transition in memory
    def push(self, *args):
        self.memory.append(Transition(*args))

    # Return a numer of random Transition from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Give the amount of stored Transitions
    def __len__(self):
        return len(self.memory)

    # reset memory
    def clear(self):
        self.memory.clear()

    # add deque to deque
    def extend(self, mem):
        self.memory.extend(mem.memory)

    def winner(self):
        size = len(self)
        i = 1
        for tran in self.memory:
            tran = tran._replace(reward=i / size)
            i += 1


# Define the Network Model
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(58 * 5 + 6, 256)
        self.bn1 = nn.BatchNorm1d(1)
        # self.layer2 = nn.Linear(512,256)
        # self.bn2 = nn.BatchNorm1d(1)
        self.layer3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(1)
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
            xi = F.relu(self.bn3(self.layer3(xi)))
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
BATCH_SIZE = 512 * 4 * 4
GAMMA = 0.96
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
TARGET_UPDATE = 1
NUM_EVAL = 600

steps_done = 0




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


# Select epsilon greedy action
def select_action(state, move_pieces):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        # Select greedy action
        return select_greedy_action(state, move_pieces)
    else:
        # returns a random movable piece from the move_pieces list. move_pieces is a list of indexes in player_pieces that can be moved
        return select_random_action(move_pieces)


# Define optimisation
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # print(transitions[0].state.shape, transitions[0].next_state.shape )
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # print(batch)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.stack(batch.next_state)

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_values = torch.cat(batch.next_q_values)

    # print("state", state_batch.shape, "action", action_batch.shape, "reward", reward_batch.shape, "non_final_next_states", non_final_next_states.shape)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(2, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    # next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # with torch.no_grad():
    #    kage = target_net(non_final_next_states).max(2)[0].detach()
    #    next_state_values[non_final_mask] = kage.flatten()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # print("hello its me")


# Evaluation
def evaluate(n_games):
    # Which players are random:

    def one_game():
        rand_players = [0, 1, 1, 1]
        random.shuffle(rand_players)
        game = ludopy.Game()

        there_is_a_winner = False
        n_round = 0
        g3_goal = 0

        while not there_is_a_winner:
            n_round +=0.25
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = game.get_observation()

            action = -1

            if len(move_pieces) > 1:  # Actual decision

                if rand_players[player_i]:  # if random
                    action, _ = select_random_action(move_pieces)
                else:
                    state = build_input(dice, player_pieces, enemy_pieces)

                    action, _ = select_greedy_action(state, torch.tensor(move_pieces))

                    # action = move_pieces[action]

            elif len(move_pieces) == 1:
                action = move_pieces[0]

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
                return [rand_players[player_i] , n_round, g3_goal]

    num_cores = multiprocessing.cpu_count()
    games = np.array(Parallel(n_jobs=num_cores)(delayed(one_game)() for i in range(n_games)))

    wins = sum(games[:, 0])
    rounds = sum(games[:, 1])/n_games
    rounds_3 = sum(games[:, 2]) / n_games
    print("")
    print("Ai:", n_games - wins, (n_games - wins) / n_games)
    print("Ra:", wins, wins / n_games)
    print("Ro:", rounds)

    return [wins, n_games - wins, rounds, rounds_3]


# def one train game


def train_game():
    rand_players = [0, 0, 0, 2]
    random.shuffle(rand_players)
    game = ludopy.Game()
    memories = [ReplayMemory(500), ReplayMemory(500), ReplayMemory(500), ReplayMemory(500)]
    #memories = ReplayMemory(2800)
    there_is_a_winner = False

    prev_state =  [None, None, None, None]
    prev_action = [None, None, None, None]
    prev_reward = [None, None, None, None]

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = game.get_observation()
        action = -1

        if rand_players[player_i] == 2:  # if semi
            _, enemy_pieces_seen_from = game.get_pieces(seen_from=player_i)
            for i in range(3):
                for piece in range(4):
                    if enemy_pieces_seen_from[i][piece] != 0 and enemy_pieces_seen_from[i][piece] <= 51:
                        enemy_pieces_seen_from[i][piece] = (enemy_pieces_seen_from[i][piece] + (13 + i * 13)) % 52
                    else:
                        enemy_pieces_seen_from[i][piece] = 0
            action = semi_int(dice, move_pieces, player_pieces, enemy_pieces_seen_from)

            _, _, _, _, _, there_is_a_winner  = game.answer_observation(action)

            if there_is_a_winner:
                return ReplayMemory(1)

        else:
            state = build_input(dice, player_pieces, enemy_pieces)

            if len(move_pieces) > 1:
                action, _ = select_action(state, torch.tensor(move_pieces))
            elif len(move_pieces) == 1:
                action = move_pieces[0]


            _, _, player_pieces_next, enemy_pieces_next, _, there_is_a_winner = game.answer_observation(action)

            if action > -1:
                # Player state
                n_home = 0
                n_in_goal = 0
                for pp in player_pieces:
                    if pp == 0:
                        n_home += 1
                    if pp == 57:
                        n_in_goal += 1

                # Player next state
                n_in_goal_next = 0
                n_home_next = 0
                for pp in player_pieces_next:
                    if pp == 0:
                        n_home_next += 1
                    if pp == 57:
                        n_in_goal_next += 1

                # Enemy state
                n_en_home = 0
                for enemy in enemy_pieces:
                    for ep in enemy:
                        if ep == 0:
                            n_en_home += 1

                # Enemy next state
                n_en_home_next = 0
                for enemy in enemy_pieces_next:
                    for ep in enemy:
                        if ep == 0:
                            n_en_home_next += 1

                reward = prev_reward[player_i]
                if n_in_goal_next == 4:
                    reward = 20             #win reward

                # next_state, _ = build_next_state(player_pieces, enemy_pieces)

                if prev_state[player_i] is not None:
                    _, next_best_q_value = select_greedy_action(state, torch.tensor(move_pieces))

                    reward = torch.tensor([reward], device=device)

                    memories[player_i].push(
                    #memories.push(
                        prev_state[player_i].to(device),
                        torch.tensor([[[prev_action[player_i]]]], device=device, dtype=torch.int64),
                        torch.tensor([next_best_q_value], device=device),
                        reward.to(device))
                # if(player_pieces_nextenemy_pieces_next.count(0))
                # prev_reward[player_i] =
                prev_state[player_i] = state
                prev_action[player_i] = action

                prev_reward[player_i] = -0.05  #step
                if n_home < n_home_next:
                    prev_reward[player_i] = -1   #suicide

                if n_en_home < n_en_home_next:
                    prev_reward[player_i] = 1  #kill

                if n_in_goal_next == 4:
                    return memories[player_i]
                    #return memories


# Training loop
#epoch

folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir("tests/{}".format(folder))

for test_i in range(4):
    num_episodes = 801
    steps_done = 800

    os.mkdir("tests/{}/{}".format(folder, test_i))

    # init DQNs
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    # Load Checkpoint
    checkpoint = torch.load("tests/2022-05-23_13-30-13/0/model_000800.state")
    policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # init opimazer and replay memory
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(80000)

    ai_win_per = 0
    for i_episode in range(800,800+num_episodes):

        if i_episode % TARGET_UPDATE == 0:
            torch.save(policy_net.state_dict(), "tests/{}/{}/model_{:06d}.state".format(folder, test_i, i_episode))

            file = open("tests/{}/{}/data.csv".format(folder,test_i), 'a')
            data = evaluate(NUM_EVAL)
            file.write("{:d},{:d},{:f},{:f},{:f},{:f},{:f}\n".format(i_episode, NUM_EVAL, data[0], data[1], data[2], data[3], ai_win_per))

            file.close()

        print("Epoch:", i_episode)
        #memory.clear()

        num_cores = multiprocessing.cpu_count()
        games_mem = Parallel(n_jobs=num_cores)(delayed(train_game)() for i in range(400))

        ai_win_count = 0
        for mem in games_mem:
            if len(mem) > 0:
                ai_win_count += 1
            memory.extend(mem)
        ai_win_per = ai_win_count/400.0
        print("Stored memory:", len(memory))
        print("Ai train win rate:", "{}%".format(ai_win_per*100))

        for n in range(20):
            optimize_model()
        steps_done += 1

        # if i_episode % TARGET_UPDATE == 0:
        #    target_net.load_state_dict(policy_net.state_dict())

