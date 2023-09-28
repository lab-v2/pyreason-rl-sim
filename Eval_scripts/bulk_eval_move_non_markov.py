import torch
import os
import sys
import json
import gym
import pyreason_gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from collections import deque
import itertools
import random
import time
import tracemalloc


OPPO_RANDOMNESS_START = 0.3
OPPO_RANDOMNESS_END = 0.05
NUM_AGENTS_TEAM = 2

class BattleField:
    def __init__(self, size, num_agents, red_poses, blue_poses, eps):
        self.size = size
        self.num_agents = num_agents
        self.team_1 = 'red_team'
        self.team_2 = 'blue_team'
        self.soldier_idx = 1    # Moves slower
        self.tank_idx = 0       # Moves faster
        self.red_poses = red_poses
        self.blue_poses = blue_poses
        self.red_poses_prev = red_poses
        self.blue_poses_prev = blue_poses
        self.blue_randomness = eps
        self.blue_pref_dir = [random.randint(0, 1) for _ in range(num_agents)]
        self.red_base = (7,0)
        self.blue_base = (0,7)
        self.max_dist = abs(self.blue_base[0] - self.red_base[0]) + abs(self.blue_base[1] - self.red_base[1])
        self.actions_prg = ['up', 'down', 'left', 'right', 'shoot_up', 'shoot_down', 'shoot_left', 'shoot_right', 'nop']
        self.actions = ['up', 'down', 'left', 'right', 'nop']
        self.mountains = [(2,3), (3,3), (2,4), (3,4), (4,4), (4,5)]
        self.winner = None
    
    def get_state(self, idx):
        state = []
        curr_pos = self.red_poses[idx]
        prev_pos = self.red_poses_prev[idx]
        state.append(curr_pos[0])
        state.append(curr_pos[1])
        state.append(prev_pos[0])
        state.append(prev_pos[1])
        # for id, red_pos in enumerate(self.red_poses):
        #     if id != idx:
        #         state.append(red_pos[0])
        #         state.append(red_pos[1])
        for blue_pos in self.blue_poses:
            state.append(blue_pos[0])
            state.append(blue_pos[1])
        for blue_pos in self.blue_poses_prev:
            state.append(blue_pos[0])
            state.append(blue_pos[1])
        return np.asarray(state,dtype=np.float32)
    
    def get_states(self):
        return [self.get_state(id) for id in range(self.num_agents)]
    
    def sample_actions(self):
        return [random.randint(0, len(self.actions)-1) for _ in range(self.num_agents)]
    
    def sample_action(self):
        return random.randint(0, len(self.actions)-1)
    
    def reset(self, size, red_poses, blue_poses, eps=None):
        self.size = size
        self.red_poses = red_poses
        self.blue_poses = blue_poses
        self.red_poses_prev = red_poses
        self.blue_poses_prev = blue_poses
        self.blue_pref_dir = [random.randint(0, 1) for _ in range(self.num_agents)]
        if eps is not None:
            self.blue_randomness = eps
        self.winner = None
        return self
    
    ## Interfaces with the gym env and tells if for
    # the given agent what is the observation of its 
    # new pos and if enemy is killed
    ## Use gym observation to update agent pos and also assign REWARDS
    def move(self, actions):
        action = {}
        ## Only need to consider Pyreason's action space here
        action['red_team'] = list(map(lambda x: self.actions_prg.index(x), actions[0]))
        action['blue_team'] = list(map(lambda x: self.actions_prg.index(x), actions[1]))

        # print(action)
        obs_a, rew, done, _, _ = env.step(action)
        # print(obs_a)
        new_red_poses = []
        new_blue_poses = []
        for idx in range(self.num_agents):
            new_red_poses.append((obs_a["red_team"][idx]["pos"][0],obs_a["red_team"][idx]["pos"][1]))
            new_blue_poses.append((obs_a["blue_team"][idx]["pos"][0],obs_a["blue_team"][idx]["pos"][1]))
        self.red_poses_prev = self.red_poses
        self.blue_poses_prev = self.blue_poses
        self.red_poses = new_red_poses
        self.blue_poses = new_blue_poses

    def validMove(self, pos, dir):
        if dir == 'nop':
            return True
        if dir == 'up':
            new_pos = (pos[0], pos[1]+1)
        elif dir == 'down':
            new_pos = (pos[0], pos[1]-1)
        elif dir == 'left':
            new_pos = (pos[0]-1, pos[1])
        elif dir == 'right':
            new_pos = (pos[0]+1, pos[1])
        else:
            new_pos = (pos[0], pos[1])
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] > self.size-1 or new_pos[1] > self.size-1:
            return False
        if new_pos in self.mountains:
            return False
        return True

    def apply_actions(self, red_actions, can_soldier_act):
        ## Get n actions for blue agents
        blue_actions = list(map(lambda id: 'nop' if not can_soldier_act and id == self.soldier_idx else self.actuate_blue(id), range(self.num_agents)))
        ## Filter actions for red agents based on validity of the action
        filtered_actions = [self.actions[ac] if self.validMove(self.red_poses[id], self.actions[ac]) else 'nop' for id, ac in enumerate(red_actions)]
        ## Tag invalid actions to penalize in reward function
        valid_actions = [self.validMove(self.red_poses[id], self.actions[ac]) for id, ac in enumerate(red_actions)]

        ## Take the actions in the gym env
        self.move([filtered_actions,blue_actions])

        ## Check if any of the blue team agents have reached red base
        blue_team_wins = any([pos == self.red_base for pos in self.blue_poses])

        ## Give rewards and also record the done flags separately
        return list(map(list, zip(*[self.get_rew(idx, valid_actions[idx], blue_team_wins) for idx in range(self.num_agents)])))

    ## AKA. REWARDS FN
    ## Remains unchanged
    def get_rew(self, idx, valid_action, blue_wins):
        
        if valid_action:
            if self.red_poses[idx] == self.blue_base:
                ## Both have reached the each other's bases
                if blue_wins:
                    self.winner = 'Tie'
                    return 50, True
                
                self.winner = 'Red'
                return 500, True
            else:
                if blue_wins:
                    self.winner = 'Blue'
                    goal_dist = (abs(self.blue_base[0] - self.red_poses[idx][0]) + abs(self.blue_base[1] - self.red_poses[idx][1])) / self.max_dist
                    return (-250 * goal_dist), True
                return -1, False
        else:
            if blue_wins:
                self.winner = 'Blue'
                goal_dist = (abs(self.blue_base[0] - self.red_poses[idx][0]) + abs(self.blue_base[1] - self.red_poses[idx][1])) / self.max_dist
                return (-250 * goal_dist), True
            return -200, False


    ## Function to only get a Blue agent's actions
    def actuate_blue(self, idx):
        if random.random() <= self.blue_randomness:
            dir = random.choice(self.actions)
            if self.validMove(self.blue_poses[idx], dir):
                return dir
            return 'nop'
        else:
            if self.blue_pref_dir[idx]:
                y_dis = self.blue_poses[idx][1] - self.red_base[1]
                if y_dis > 0:
                    return 'down'
                elif y_dis < 0:
                    return 'up'
                else:
                    x_dis = self.blue_poses[idx][0] - self.red_base[0]
                    if x_dis > 0:
                        return 'left'
                    elif x_dis < 0:
                        return 'right'
            else:
                x_dis = self.blue_poses[idx][0] - self.red_base[0]
                if x_dis > 0:
                    return 'left'
                elif x_dis < 0:
                    return 'right'
                else:
                    y_dis = self.blue_poses[idx][1] - self.red_base[1]
                    if y_dis > 0:
                        return 'down'
                    elif y_dis < 0:
                        return 'up'

class Network(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()

        in_features = int(np.prod(env.get_state(0).shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, len(env.actions))
        )
    
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


def reset_env(env, field, eps=None):
    obs_a, _ = env.reset()
    size = 8
    new_red_poses = []
    new_blue_poses = []
    for idx in range(field.num_agents):
        new_red_poses.append((obs_a["red_team"][idx]["pos"][0],obs_a["red_team"][idx]["pos"][1]))
        new_blue_poses.append((obs_a["blue_team"][idx]["pos"][0],obs_a["blue_team"][idx]["pos"][1]))

    field = field.reset(size, new_red_poses, new_blue_poses, eps)

    obs = field.get_states()
    return field, obs

env = gym.make('PyReasonGridWorld-v0', num_agents_per_team=NUM_AGENTS_TEAM)
field = BattleField(None, NUM_AGENTS_TEAM, None, None, OPPO_RANDOMNESS_START)
field, obs = reset_env(env, field)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tank_net = Network(field)
soldier_net = Network(field)

DATA_PATH = f"./success_non_markov_{NUM_AGENTS_TEAM}/"
dir_files = os.listdir(DATA_PATH)

MAX_EP = 500
MAX_EP_STEPS = 200

eval_step_metrics = {}

for y in dir_files:
    if y.endswith(".pth") and y.startswith("dqn_Move-Multi-NonM-Soldier_1_Aug_12PM_"):
        eval_step = y.split("_")[-1].split(".")[0][1:]
        eval_step = int(eval_step)
        if eval_step % 16000 == 0:
            print("Started eval for step:", eval_step)
            print("Loading Soldier NN from path", DATA_PATH + y)
            soldier_net.load_state_dict(torch.load(DATA_PATH + y))
            soldier_net.eval()
            t = f'dqn_Move-Multi-NonM-Tank_1_Aug_12PM_s{eval_step}.pth'
            print("Loading Tank NN from path", DATA_PATH + t)
            tank_net.load_state_dict(torch.load(DATA_PATH + t))
            tank_net.eval()

            field, obs = reset_env(env, field)

            can_soldier_act = False
            soldier_rew = None
            
            soldier_reward_buffer = deque([0.0], maxlen=100)
            episode_reward_s = 0.0
            reward_buffer = deque([0.0], maxlen=100)
            episode_reward = 0.0
            episode_count = 0
            episode_steps = 0
            terminated_ep_count = 0
            wins = []
            avg_episode_len = []
            avg_episode_len_non_term = []
            run_times = []
            run_times_non_term = []
            mem_usages = []
            mem_usages_non_term = []
            terminated = False
            tracemalloc.start()
            start = time.time()

            ## Playing loop
            for step in itertools.count():
                # print("Step:",episode_steps)
                # sys.stdout.flush()
                can_soldier_act = not can_soldier_act

                actions = []
                for id in range(field.num_agents):
                    if id == field.tank_idx:
                        action = tank_net.act(obs[id])
                    elif id == field.soldier_idx:
                        action = soldier_net.act(obs[id])

                    actions.append(action)

                if not can_soldier_act:
                    actions[1] = 4 ## Take a nop if not soldier's time to act

                rews_dones = field.apply_actions(actions, can_soldier_act)

                new_obs = field.get_states()

                if can_soldier_act:
                    ## Save any immediate reward -> invalid actions only when an action is taken
                    soldier_rew = rews_dones[0][field.soldier_idx]

                obs = new_obs

                episode_reward += rews_dones[0][field.tank_idx]

                episode_steps += 1

                if episode_steps > MAX_EP_STEPS:
                    terminated = True

                if any(rews_dones[1]) or terminated:
                    print("Episode done!!")
                    sys.stdout.flush()
                    if rews_dones[1][field.soldier_idx]:
                        episode_reward += rews_dones[0][field.soldier_idx]
                        episode_reward_s += rews_dones[0][field.soldier_idx]

                    reward_buffer.append(episode_reward)
                    soldier_reward_buffer.append(episode_reward_s)
                    
                    time_elapsed = time.time() - start
                    _, mem_peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    run_times.append(time_elapsed)
                    mem_usages.append(mem_peak)
                    avg_episode_len.append(episode_steps)

                    if terminated:
                        wins.append('Blue')
                        terminated_ep_count += 1
                    else:
                        wins.append(field.winner)
                        run_times_non_term.append(time_elapsed)
                        mem_usages_non_term.append(mem_peak)
                        avg_episode_len_non_term.append(episode_steps)

                    field, obs = reset_env(env, field)
                    episode_reward = 0.0
                    episode_reward_s = 0.0
                    episode_count += 1
                    episode_steps = 0
                    terminated = False
                    can_soldier_act = False
                    soldier_rew = None

                    if episode_count >= MAX_EP:
                        avg_ep_len = float(np.mean(avg_episode_len))
                        avg_ep_rew = float(np.mean(reward_buffer))
                        avg_ep_sol_rew = float(np.mean(soldier_reward_buffer))
                        win_per = wins.count('Red')
                        max_mem = float(np.max(mem_usages))
                        run_time_mean = float(np.mean(run_times))
                        run_time_med = float(np.median(run_times))
                        try:
                            max_mem_non_term = float(np.max(mem_usages_non_term))
                        except ValueError:
                            max_mem_non_term = 0.0
                            pass
                        try:
                            run_time_mean_non_term = float(np.mean(run_times_non_term))
                        except ValueError:
                            run_time_mean_non_term = 0.0
                            pass
                        try:
                            run_time_med_non_term = float(np.median(run_times_non_term))
                        except ValueError:
                            run_time_med_non_term = 0.0
                            pass
                        try:
                            avg_ep_len_non_term = float(np.mean(avg_episode_len_non_term))
                        except ValueError:
                            avg_ep_len_non_term = 0.0
                            pass
                        print('Avg Episode Reward:', avg_ep_rew, " points")
                        print('Avg Episode Len:', avg_ep_len, " steps")
                        print('Win Percent (Red):', win_per)
                        eval_step_metrics[eval_step] = {
                            "avg_ep_len": avg_ep_len,
                            "avg_rew": avg_ep_rew,
                            "avg_sol_rew": avg_ep_sol_rew,
                            "win_per": win_per,
                            "max_mem": max_mem,
                            "run_time_mean": run_time_mean,
                            "run_time_median": run_time_med,
                            "avg_ep_len_non_term": avg_ep_len_non_term,
                            "max_mem_non_term": max_mem_non_term,
                            "run_time_mean_non_term": run_time_mean_non_term,
                            "run_time_median_non_term": run_time_med_non_term,
                            "terminated_ep_count": terminated_ep_count
                        }
                        sys.stdout.flush()
                        break
                    continue
                
                if not any(rews_dones[1]) and not can_soldier_act:
                    ## This is when the effect of the action takes place so save this in replay buffer
                    episode_reward += soldier_rew
                    episode_reward_s += soldier_rew

print(eval_step_metrics)
with open(f'non_markov_success_eval_{NUM_AGENTS_TEAM}.json', 'w') as fp:
    json.dump(eval_step_metrics, fp, indent=4)