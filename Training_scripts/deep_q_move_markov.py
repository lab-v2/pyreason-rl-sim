import torch
import gym
import pyreason_gym
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
from torch import nn
from collections import deque
import itertools
import random
from datetime import datetime

MAX_STEPS = 800000
BATCH_SIZE = 128
## always [0, 1),  
# A lower γ makes rewards from the uncertain far future
# less important for our agent than the ones in the
# near future that it can be fairly confident about.
# It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
GAMMA = 0.99
EPS_VAL_START = 1.0
EPS_VAL_END = 0.02
EPS_DECAY_START = (MAX_STEPS * 2)/8
EPS_DECAY_END = (MAX_STEPS * 7)/8

OPPO_RANDOMNESS_START = 0.3
OPPO_RANDOMNESS_END = 0.05

BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
UPDATE_FREQ = 2000
EVAL_FREQ = 4000
LR = 5e-4
NUM_AGENTS_TEAM = 1


TIME_STR = datetime.now().strftime("%-d_%b_%-I%p")

class BattleField:
    def __init__(self, size, num_agents, red_poses, blue_poses, eps):
        self.size = size
        self.num_agents = num_agents
        self.team_1 = 'red_team'
        self.team_2 = 'blue_team'
        self.red_poses = red_poses
        self.blue_poses = blue_poses
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
        state.append(curr_pos[0])
        state.append(curr_pos[1])
        # for id, red_pos in enumerate(self.red_poses):
        #     if id != idx:
        #         state.append(red_pos[0])
        #         state.append(red_pos[1])
        for blue_pos in self.blue_poses:
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

    def apply_actions(self, red_actions):
        ## Get n actions for blue agents
        blue_actions = list(map(lambda id: self.actuate_blue(id), range(self.num_agents)))
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
        if random.random() <= max(self.blue_randomness, OPPO_RANDOMNESS_END):
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
field = BattleField(None, NUM_AGENTS_TEAM, None, None, min(OPPO_RANDOMNESS_START, EPS_VAL_START))
field, obs = reset_env(env, field)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU support? ", torch.cuda.is_available())

print("MAX_STEPS", MAX_STEPS)
print("GAMMA", GAMMA)
print("EPS_VAL_START", EPS_VAL_START)
print("EPS_VAL_END", EPS_VAL_END)
print("OPPO_RANDOMNESS_START", OPPO_RANDOMNESS_START)
print("UPDATE_FREQ", UPDATE_FREQ)
print("EPS_DECAY_START", EPS_DECAY_START)
print("EPS_DECAY_END", EPS_DECAY_END)
print("NUM_AGENTS_TEAM", NUM_AGENTS_TEAM)
print("TIME ", TIME_STR)

online_net = Network(field)
target_net = Network(field)

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

for _ in range(MIN_REPLAY_SIZE):
    actions = field.sample_actions()

    rews_dones = field.apply_actions(actions)

    new_obs = field.get_states()

    for idx in range(field.num_agents):
        transition = (obs[idx], actions[idx], rews_dones[0][idx], rews_dones[1][idx], new_obs[idx])
        replay_buffer.append(transition)

    obs = new_obs

    if any(rews_dones[1]):
        field, obs = reset_env(env, field)

field, obs = reset_env(env, field)

## Training loop
for step in itertools.count():
    epsilon = np.interp(step, [EPS_DECAY_START, EPS_DECAY_END], [EPS_VAL_START, EPS_VAL_END])

    actions = []
    for id in range(field.num_agents):
        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = field.sample_action()
        else:
            action = online_net.act(obs[id])
        actions.append(action)

    rews_dones = field.apply_actions(actions)

    new_obs = field.get_states()

    for idx in range(field.num_agents):
        transition = (obs[idx], actions[idx], rews_dones[0][idx], rews_dones[1][idx], new_obs[idx])
        replay_buffer.append(transition)

    obs = new_obs

    episode_reward += sum(rews_dones[0])

    if any(rews_dones[1]):
        field, obs = reset_env(env, field)

        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obsers = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obsers = np.asarray([t[4] for t in transitions])

    obsers_t = torch.as_tensor(obsers, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obsers_t = torch.as_tensor(new_obsers, dtype=torch.float32)

    target_q_values = target_net(new_obsers_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    q_values = online_net(obsers_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step != 0 and step % UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    if step != 0 and step % EVAL_FREQ == 0:
        print()
        print('Step', step)
        print('Avg Reward:', np.mean(reward_buffer))
        sys.stdout.flush()
        ## the target/online net for later usage
        torch.save(target_net.state_dict(), f'dqn_Move-Multi-1-Rich_{TIME_STR}_s{step}.pth')
    
    if step == MAX_STEPS:
        break
