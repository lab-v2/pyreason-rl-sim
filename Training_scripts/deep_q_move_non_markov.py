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

MAX_STEPS = 1600000
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
MIN_REPLAY_SIZE = 4000
UPDATE_FREQ = 2000
EVAL_FREQ = 16000
LR = 5e-4
NUM_AGENTS_TEAM = 2


TIME_STR = datetime.now().strftime("%-d_%b_%-I%p")

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

## Tank NNs
online_net = Network(field)
target_net = Network(field)

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

replay_buffer = deque(maxlen=BUFFER_SIZE)

## Soldier NNs
online_net_s = Network(field)
target_net_s = Network(field)

target_net_s.load_state_dict(online_net_s.state_dict())
optimizer_s = torch.optim.Adam(online_net_s.parameters(), lr=LR)

replay_buffer_s = deque(maxlen=BUFFER_SIZE)


reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

soldier_reward_buffer = deque([0.0], maxlen=100)
episode_reward_s = 0.0

can_soldier_act = False
soldier_act = None
soldier_obs = None
soldier_rew = None
soldier_new_obs = None

for _ in range(MIN_REPLAY_SIZE):
    can_soldier_act = not can_soldier_act

    actions = field.sample_actions()

    if can_soldier_act:
        ## Save actions only when they are taken and also the obs basis which the actions were taken
        soldier_act = actions[field.soldier_idx]
        soldier_obs = obs[field.soldier_idx]
    else:
        actions[field.soldier_idx] = 4 ## Take a nop if not soldier's time to act

    rews_dones = field.apply_actions(actions, can_soldier_act)

    new_obs = field.get_states()

    if can_soldier_act:
        ## Save any immediate reward -> invalid actions only when an action is taken
        soldier_rew = rews_dones[0][field.soldier_idx]

    ## Making this explicit
    if not can_soldier_act:
        ## This is when the effect of the action takes place so save this new_obs
        soldier_new_obs = new_obs[field.soldier_idx]

    transition = (obs[field.tank_idx], actions[field.tank_idx], rews_dones[0][field.tank_idx], rews_dones[1][field.tank_idx], new_obs[field.tank_idx])
    replay_buffer.append(transition)

    obs = new_obs

    if any(rews_dones[1]):
        ## Any time an episode concludes(done) add the concluded state as the transition
        # for slow moving soldier or else this will be lost (only if the concluded state occurs for the soldier)
        # if not a final state for the soldier then discard the transition because the final state is not known
        if rews_dones[1][field.soldier_idx]:
            transition_s = (soldier_obs, soldier_act, rews_dones[0][field.soldier_idx], True, new_obs[field.soldier_idx])
            replay_buffer_s.append(transition_s)

        field, obs = reset_env(env, field)
        can_soldier_act = False
        soldier_act = None
        soldier_obs = None
        soldier_rew = None
        soldier_new_obs = None

        continue

    if not can_soldier_act:
        ## This is when the effect of the action takes place so save this in replay buffer
        transition_s = (soldier_obs, soldier_act, soldier_rew, False, soldier_new_obs)
        replay_buffer_s.append(transition_s)


field, obs = reset_env(env, field)

can_soldier_act = False
soldier_act = None
soldier_obs = None
soldier_rew = None
soldier_new_obs = None

## Training loop
for step in itertools.count():
    epsilon = np.interp(step, [EPS_DECAY_START, EPS_DECAY_END], [EPS_VAL_START, EPS_VAL_END])

    can_soldier_act = not can_soldier_act

    actions = []
    for id in range(field.num_agents):
        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = field.sample_action()
        else:
            if id == field.tank_idx:
                action = online_net.act(obs[id])
            elif id == field.soldier_idx:
                action = online_net_s.act(obs[id])

        actions.append(action)
    
    if can_soldier_act:
        ## Save actions only when they are taken and also the obs basis which the actions were taken
        soldier_act = actions[field.soldier_idx]
        soldier_obs = obs[field.soldier_idx]
    else:
        actions[1] = 4 ## Take a nop if not soldier's time to act

    rews_dones = field.apply_actions(actions, can_soldier_act)

    new_obs = field.get_states()

    if can_soldier_act:
        ## Save any immediate reward -> invalid actions only when an action is taken
        soldier_rew = rews_dones[0][field.soldier_idx]

    if not can_soldier_act:
        ## This is when the effect of the action takes place so save this new_obs
        soldier_new_obs = new_obs[field.soldier_idx]

    transition = (obs[field.tank_idx], actions[field.tank_idx], rews_dones[0][field.tank_idx], rews_dones[1][field.tank_idx], new_obs[field.tank_idx])
    replay_buffer.append(transition)

    obs = new_obs

    episode_reward += rews_dones[0][field.tank_idx]

    if any(rews_dones[1]):
        ## Any time an episode concludes(done) add the concluded state as the transition
        # for slow moving soldier or else this will be lost (only if the concluded state occurs for the soldier)
        # if not a final state for the soldier then discard the transition because the final state is not known
        if rews_dones[1][field.soldier_idx]:
            transition_s = (soldier_obs, soldier_act, rews_dones[0][field.soldier_idx], True, new_obs[field.soldier_idx])
            replay_buffer_s.append(transition_s)
            episode_reward += rews_dones[0][field.soldier_idx]
            episode_reward_s += rews_dones[0][field.soldier_idx]

        field, obs = reset_env(env, field)
        can_soldier_act = False
        soldier_act = None
        soldier_obs = None
        soldier_rew = None
        soldier_new_obs = None

        reward_buffer.append(episode_reward)
        soldier_reward_buffer.append(episode_reward_s)
        episode_reward = 0.0
        episode_reward_s = 0.0

    ## Episode is not concluded and the soldier hasn't acted in this step
    ## Otherwise ignore the half completed action since effect is not known
    if not any(rews_dones[1]) and not can_soldier_act:
        ## This is when the effect of the action takes place so save this in replay buffer
        transition_s = (soldier_obs, soldier_act, soldier_rew, False, soldier_new_obs)
        replay_buffer_s.append(transition_s)
        episode_reward += soldier_rew
        episode_reward_s += soldier_rew

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

    ## Only update for solder every alternate step
    if not can_soldier_act:
        transitions_s = random.sample(replay_buffer_s, BATCH_SIZE)

        obsers_s = np.asarray([t[0] for t in transitions_s])
        actions_s = np.asarray([t[1] for t in transitions_s])
        rewards_s = np.asarray([t[2] for t in transitions_s])
        dones_s = np.asarray([t[3] for t in transitions_s])
        new_obsers_s = np.asarray([t[4] for t in transitions_s])

        obsers_s_t = torch.as_tensor(obsers_s, dtype=torch.float32)
        actions_s_t = torch.as_tensor(actions_s, dtype=torch.int64).unsqueeze(-1)
        rewards_s_t = torch.as_tensor(rewards_s, dtype=torch.float32).unsqueeze(-1)
        dones_s_t = torch.as_tensor(dones_s, dtype=torch.float32).unsqueeze(-1)
        new_obsers_s_t = torch.as_tensor(new_obsers_s, dtype=torch.float32)

        target_q_values_s = target_net_s(new_obsers_s_t)
        max_target_q_values_s = target_q_values_s.max(dim=1, keepdim=True)[0]

        targets_s = rewards_s_t + GAMMA * (1 - dones_s_t) * max_target_q_values_s

        q_values_s = online_net_s(obsers_s_t)

        action_q_values_s = torch.gather(input=q_values_s, dim=1, index=actions_s_t)

        loss_s = nn.functional.smooth_l1_loss(action_q_values_s, targets_s)

        optimizer_s.zero_grad()
        loss_s.backward()
        optimizer_s.step()

    if step != 0 and step % UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    if step != 0 and step % (UPDATE_FREQ*2) == 0:
        target_net_s.load_state_dict(online_net_s.state_dict())

    if step != 0 and step % EVAL_FREQ == 0:
        print()
        print('Step', step)
        print('Avg Reward:', np.mean(reward_buffer))
        print('Avg Soldier Reward:', np.mean(soldier_reward_buffer))
        sys.stdout.flush()
        ## the target/online net for later usage
        torch.save(target_net.state_dict(), f'dqn_Move-Multi-NonM-Tank_{TIME_STR}_s{step}.pth')
        torch.save(target_net_s.state_dict(), f'dqn_Move-Multi-NonM-Soldier_{TIME_STR}_s{step}.pth')
    
    if step == MAX_STEPS:
        break
