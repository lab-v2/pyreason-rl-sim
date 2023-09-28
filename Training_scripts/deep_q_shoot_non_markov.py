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

MAX_STEPS = 2400000
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
EVAL_FREQ = 24000
LR = 5e-4
NUM_AGENTS_TEAM = 2


TIME_STR = datetime.now().strftime("%-d_%b_%-I%p")

class BattleField:
    def __init__(self, size, num_agents, red_poses, blue_poses, eps):
        self.size = size
        self.num_agents = num_agents
        self.team_red = 'red_team'
        self.team_blue = 'blue_team'
        self.soldier_idx = 1    # Moves slower
        self.tank_idx = 0       # Moves faster
        self.ammo = {
            f"{self.team_red}": [3] * num_agents,
            f"{self.team_blue}": [3] * num_agents,
        }
        self.killed = {
            f"{self.team_red}": [False] * num_agents,
            f"{self.team_blue}": [False] * num_agents,
        }
        self.red_poses = red_poses
        self.blue_poses = blue_poses
        self.red_poses_prev = red_poses
        self.blue_poses_prev = blue_poses
        self.bullet_cnt = 0
        self.bullet_cnt_prev = 0
        self.near_bullet_pos = [(0,0)] * num_agents
        self.near_bullet_pos_prev = [(0,0)] * num_agents
        self.red_agent_kill_cnt = [0] * num_agents
        self.blue_randomness = eps
        self.blue_pref_dir = [random.randint(0, 1) for _ in range(num_agents)]
        self.red_base = (7,0)
        self.blue_base = (0,7)
        self.max_dist = abs(self.blue_base[0] - self.red_base[0]) + abs(self.blue_base[1] - self.red_base[1])
        self.actions_prg = ['up', 'down', 'left', 'right', 'shoot_up', 'shoot_down', 'shoot_left', 'shoot_right', 'nop']
        self.actions = ['up', 'down', 'left', 'right', 'shoot_up', 'shoot_down', 'shoot_left', 'shoot_right', 'nop']
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
        state.append(self.bullet_cnt)
        state.append(self.bullet_cnt_prev)
        state.append(self.near_bullet_pos[idx][0])
        state.append(self.near_bullet_pos[idx][1])
        state.append(self.near_bullet_pos_prev[idx][0])
        state.append(self.near_bullet_pos_prev[idx][1])
        state.append(self.ammo[self.team_red][idx])
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
        self.near_bullet_pos = [(0,0)] * self.num_agents
        self.near_bullet_pos_prev = [(0,0)] * self.num_agents
        self.red_agent_kill_cnt = [0] * self.num_agents
        self.bullet_cnt = 0
        self.bullet_cnt_prev = 0
        self.blue_pref_dir = [random.randint(0, 1) for _ in range(self.num_agents)]
        if eps is not None:
            self.blue_randomness = eps
        self.ammo = {
            f"{self.team_red}": [3] * self.num_agents,
            f"{self.team_blue}": [3] * self.num_agents,
        }
        self.killed = {
            f"{self.team_red}": [False] * self.num_agents,
            f"{self.team_blue}": [False] * self.num_agents,
        }
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

        # print()
        obs_a, rew, done, _, _ = env.step(action)
        # print(obs_a)

        ## check the taken actions and decrement the ammo
        for idx, ac in enumerate(actions[0]):
            if ac in ['shoot_up','shoot_down','shoot_left','shoot_right']:
                self.ammo[self.team_red][idx] -= 1
        for idx, ac in enumerate(actions[1]):
            if ac in ['shoot_up','shoot_down','shoot_left','shoot_right']:
                self.ammo[self.team_blue][idx] -= 1

        new_red_poses = []
        new_blue_poses = []
        for idx in range(self.num_agents):
            new_red_poses.append((obs_a["red_team"][idx]["pos"][0],obs_a["red_team"][idx]["pos"][1]))
            new_blue_poses.append((obs_a["blue_team"][idx]["pos"][0],obs_a["blue_team"][idx]["pos"][1]))
        self.red_poses_prev = self.red_poses
        self.blue_poses_prev = self.blue_poses
        self.red_poses = new_red_poses
        self.blue_poses = new_blue_poses

        ## Check which agents have been killed
        ## Also see if any blue agent got killed in this env-step
        blue_killed_now = [False] * self.num_agents
        for idx in range(self.num_agents):
            if obs_a[self.team_red][idx]["health"][0] == 0:
                self.killed[self.team_red][idx] = True
        for idx in range(self.num_agents):
            if obs_a[self.team_blue][idx]["health"][0] == 0:
                if not self.killed[self.team_blue][idx]:
                    blue_killed_now[idx] = True
                self.killed[self.team_blue][idx] = True

        ## Extract bullet count
        self.bullet_cnt_prev = self.bullet_cnt
        self.bullet_cnt = len(obs_a["blue_bullets"])
        ## Extract nearest bullet for each agent
        for idx in range(self.num_agents):
            min_dis = 999999
            nearest_bullet = None
            ## bullet = {'pos': array([6, 1]), 'dir': 0}
            if self.bullet_cnt == 1:
                nearest_bullet = obs_a["blue_bullets"][0]
            elif self.bullet_cnt > 1:
                for bullet in obs_a["blue_bullets"]:
                    dis = abs(obs_a["red_team"][idx]["pos"][0] - bullet["pos"][0]) + abs(obs_a["red_team"][idx]["pos"][1] - bullet["pos"][1])
                    if dis < min_dis:
                        nearest_bullet = bullet

            if nearest_bullet is None:
                self.near_bullet_pos_prev[idx] = self.near_bullet_pos[idx]
                self.near_bullet_pos[idx] = (0,0)
            else:
                self.near_bullet_pos_prev[idx] = self.near_bullet_pos[idx]
                self.near_bullet_pos[idx] = (nearest_bullet["pos"][0], nearest_bullet["pos"][1])
        
        ## Attribute which agent killed (if any) blue agents
        for b_idx, killed in enumerate(blue_killed_now):
            if killed:
                for r_idx in range(self.num_agents):
                    if (b_idx + 1) in obs_a[self.team_red][r_idx]['killed']:
                        self.red_agent_kill_cnt[r_idx] += 1
                        break


    def validMove(self, pos, dir, agent_team, idx):
        if dir == 'nop':
            return True
        if dir in ['shoot_up','shoot_down','shoot_left','shoot_right']:
            if self.ammo[agent_team][idx] > 0:
                return True
            return False
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
        filtered_actions = [self.actions[ac] if self.validMove(self.red_poses[id], self.actions[ac], self.team_red, id) else 'nop' for id, ac in enumerate(red_actions)]
        ## Tag invalid actions to penalize in reward function
        valid_actions = [self.validMove(self.red_poses[id], self.actions[ac], self.team_red, id) for id, ac in enumerate(red_actions)]

        ## Take the actions in the gym env
        self.move([filtered_actions,blue_actions])

        ## Check if any of the blue team agents have reached red base
        blue_team_wins = any([pos == self.red_base for pos in self.blue_poses])

        ## Give rewards and also record the done flags separately and supply original action taken
        return list(map(list, zip(*[self.get_rew(idx, valid_actions[idx], blue_team_wins, self.actions[red_actions[idx]]) for idx in range(self.num_agents)])))

    ## AKA. REWARDS FN
    def get_rew(self, idx, valid_action, blue_wins, action_taken):
        ## This is continous rew so ignore after using it once!
        if self.killed[self.team_red][idx]:
            return -400, False
        shooting_rew = 0
        ## 400 points for each kill
        if self.red_agent_kill_cnt[idx] > 0:
            shooting_rew = 400 * self.red_agent_kill_cnt[idx]
            self.red_agent_kill_cnt[idx] = 0
        if valid_action:
            if self.red_poses[idx] == self.blue_base:
                ## Both have reached the each other's bases
                if blue_wins:
                    self.winner = 'Tie'
                    return (50 + shooting_rew), True
                
                self.winner = 'Red'
                return (500 + shooting_rew), True
            else:
                if blue_wins:
                    self.winner = 'Blue'
                    goal_dist = (abs(self.blue_base[0] - self.red_poses[idx][0]) + abs(self.blue_base[1] - self.red_poses[idx][1])) / self.max_dist
                    return ((-250 * goal_dist) + shooting_rew), True
                return (-1 + shooting_rew), False
        else:
            if blue_wins:
                self.winner = 'Blue'
                goal_dist = (abs(self.blue_base[0] - self.red_poses[idx][0]) + abs(self.blue_base[1] - self.red_poses[idx][1])) / self.max_dist
                return ((-250 * goal_dist) + shooting_rew), True
            ## Penalize a bit less for the shooting
            if action_taken in ['shoot_up','shoot_down','shoot_left','shoot_right']:
                return (-10 + shooting_rew), False
            return (-200 + shooting_rew), False

    ## Function to only get a Blue agent's actions
    ## TODO: don't just shoot at the red agent with same index, but rather one closest
    def actuate_blue(self, idx):
        if random.random() <= self.blue_randomness:
            dir = random.choice(['up', 'down', 'left', 'right', 'nop'])
            if self.validMove(self.blue_poses[idx], dir, self.team_blue, idx):
                return dir
            return 'nop'
        else:
            ## Shoot if able to otherwise move greedily
            if self.ammo[self.team_blue][idx] > 0 and (self.blue_poses[idx][0] - self.red_poses[idx][0]) == 0:
                y_dis = self.blue_poses[idx][1] - self.red_poses[idx][1]
                if y_dis > 0:
                    return 'shoot_down'
                else:
                    return 'shoot_up'
            elif self.ammo[self.team_blue][idx] > 0 and (self.blue_poses[idx][1] - self.red_poses[idx][1]) == 0:
                x_dis = self.blue_poses[idx][0] - self.red_poses[idx][0]
                if x_dis > 0:
                    return 'shoot_left'
                else:
                    return 'shoot_right'
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
                            return 'nop'
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
                        else:
                            return 'nop'

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

env = gym.make('PyReasonGridWorld-v0', num_agents_per_team=NUM_AGENTS_TEAM, render_mode='human')
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
soldier_died = None

for _ in range(MIN_REPLAY_SIZE):
    can_soldier_act = not can_soldier_act

    actions = field.sample_actions()

    already_killed = [False] * field.num_agents
    for id in range(field.num_agents):
        if field.killed[field.team_red][id]:
            already_killed[id] = True
            actions[id] = 8 ## Take a nop if agent is dead

    if can_soldier_act:
        ## Save actions only when they are taken and also the obs basis which the actions were taken
        soldier_act = actions[field.soldier_idx]
        soldier_obs = obs[field.soldier_idx]
    else:
        actions[field.soldier_idx] = 8 ## Take a nop if not soldier's time to act

    # print()
    # print()
    # print(actions)
    # print(can_soldier_act)
    rews_dones = field.apply_actions(actions, can_soldier_act)

    # print()
    # print(rews_dones)

    new_obs = field.get_states()

    if can_soldier_act:
        ## Save any immediate reward
        soldier_rew = rews_dones[0][field.soldier_idx]
        if not already_killed[field.soldier_idx] and field.killed[field.team_red][field.soldier_idx]:
            soldier_died = True

    ## Making this explicit, This is when the effect of the action takes place so save this new_obs
    if not can_soldier_act:
        soldier_new_obs = new_obs[field.soldier_idx]
        ## Check if the agent has got any new reward from killing another agent or because it was killed
        ## Ignore if the rew is -1 for the 'nop' taken during this turn
        if rews_dones[0][field.soldier_idx] != -1:
            soldier_rew = rews_dones[0][field.soldier_idx]
        if not already_killed[field.soldier_idx] and field.killed[field.team_red][field.soldier_idx]:
            soldier_died = True

    ## Don't add the transition if agent is already killed and did not take an action in the previous step
    if not already_killed[field.tank_idx]:
        ## If the agent died in the very previous turn then use the reward and mark as episode done for the agent only
        killed_now = field.killed[field.team_red][field.tank_idx]
        ## If the agent killed the last blue agent then mark done
        last_kill = (all(field.killed[field.team_blue]) and rews_dones[0][field.tank_idx] > 0)
        transition = (obs[field.tank_idx], actions[field.tank_idx], rews_dones[0][field.tank_idx], killed_now or rews_dones[1][field.tank_idx] or last_kill, new_obs[field.tank_idx])
        replay_buffer.append(transition)

    obs = new_obs

    ## Conclude an episode only when someone reaches other's base, don't stop the episode if any one agent dies
    ## only conclude if all the red or blue agents are dead
    if any(rews_dones[1]) or all(field.killed[field.team_red]) or all(field.killed[field.team_blue]):
        ## Any time an episode concludes(done) add the concluded state as the transition
        # for slow moving soldier or else this will be lost (only if the concluded state occurs for the soldier)
        # if not a final state for the soldier then discard the transition because the final state is not known
        # Concluded state occurs when
        # 1. soldier reaches oppo base
        # 2. soldier is killed as the last agent
        # 3. soldier kills the last oppo agent
        if (not can_soldier_act and not already_killed[field.soldier_idx]) or rews_dones[1][field.soldier_idx] or (all(field.killed[field.team_red]) and soldier_died) or (all(field.killed[field.team_blue]) and soldier_rew > 0):
            transition_s = (soldier_obs, soldier_act, soldier_rew, soldier_died or rews_dones[1][field.soldier_idx] or (all(field.killed[field.team_blue]) and soldier_rew > 0), new_obs[field.soldier_idx])
            replay_buffer_s.append(transition_s)

        field, obs = reset_env(env, field)
        can_soldier_act = False
        soldier_act = None
        soldier_obs = None
        soldier_rew = None
        soldier_new_obs = None
        soldier_died = None

        continue
    
    ## Put it afterwards because there would be double attribution of the transition for soldier otherwise
    ## Don't add the transition if agent is already killed and did not take an action in the previous step
    if not can_soldier_act and (not already_killed[field.soldier_idx] or soldier_died):
        ## This is when the effect of the action takes place so save this in replay buffer
        ## If the agent died in the very previous turn then use the reward and mark as episode done for the agent only
        transition_s = (soldier_obs, soldier_act, soldier_rew, soldier_died or False, soldier_new_obs)
        replay_buffer_s.append(transition_s)
        if soldier_died:
            soldier_died = None


field, obs = reset_env(env, field)

env_reset = True
can_soldier_act = False
soldier_act = None
soldier_obs = None
soldier_rew = None
soldier_new_obs = None
## Flag to check if soldier died in between obs
soldier_died = None

## Training loop
for step in itertools.count():
    epsilon = np.interp(step, [EPS_DECAY_START, EPS_DECAY_END], [EPS_VAL_START, EPS_VAL_END])

    can_soldier_act = not can_soldier_act
    env_reset = False
    actions = []

    already_killed = [False] * field.num_agents
    for id in range(field.num_agents):
        if not field.killed[field.team_red][id]:
            rnd_sample = random.random()

            if rnd_sample <= epsilon:
                action = field.sample_action()
            else:
                if id == field.tank_idx:
                    action = online_net.act(obs[id])
                elif id == field.soldier_idx:
                    action = online_net_s.act(obs[id])
        else:
            already_killed[id] = True
            action = 8 ## Take a nop if agent is dead

        actions.append(action)
    
    if can_soldier_act:
        ## Save actions only when they are taken and also the obs basis which the actions were taken
        soldier_act = actions[field.soldier_idx]
        soldier_obs = obs[field.soldier_idx]
    else:
        actions[field.soldier_idx] = 8 ## Take a nop if not soldier's time to act

    rews_dones = field.apply_actions(actions, can_soldier_act)

    new_obs = field.get_states()

    if can_soldier_act:
        ## Save any immediate reward
        soldier_rew = rews_dones[0][field.soldier_idx]
        if not already_killed[field.soldier_idx] and field.killed[field.team_red][field.soldier_idx]:
            soldier_died = True

    ## Making this explicit, This is when the effect of the action takes place so save this new_obs
    if not can_soldier_act:
        soldier_new_obs = new_obs[field.soldier_idx]
        ## Check if the agent has got any new reward from killing another agent or because it was killed
        ## Ignore if the rew is -1 for the 'nop' taken during this turn
        if rews_dones[0][field.soldier_idx] != -1:
            soldier_rew = rews_dones[0][field.soldier_idx]
        if not already_killed[field.soldier_idx] and field.killed[field.team_red][field.soldier_idx]:
            soldier_died = True

    ## Don't add the transition if agent is already killed and did not take an action in the previous step
    if not already_killed[field.tank_idx]:
        ## If the agent died in the very previous turn then use the reward and mark as episode done for the agent only
        killed_now = field.killed[field.team_red][field.tank_idx]
        ## If the agent killed the last blue agent then mark done
        last_kill = (all(field.killed[field.team_blue]) and rews_dones[0][field.tank_idx] > 0)
        transition = (obs[field.tank_idx], actions[field.tank_idx], rews_dones[0][field.tank_idx], killed_now or rews_dones[1][field.tank_idx] or last_kill, new_obs[field.tank_idx])
        replay_buffer.append(transition)

    obs = new_obs

    episode_reward += rews_dones[0][field.tank_idx] if not already_killed[field.tank_idx] else 0

    ## Conclude an episode only when someone reaches other's base, don't stop the episode if any one agent dies
    ## only conclude if all the red or blue agents are dead
    if any(rews_dones[1]) or all(field.killed[field.team_red]) or all(field.killed[field.team_blue]):
        ## Any time an episode concludes(done) add the concluded state as the transition
        # for slow moving soldier or else this will be lost (only if the concluded state occurs for the soldier)
        # if not a final state for the soldier then discard the transition because the final state is not known
        # Concluded state occurs when
        # 1. soldier reaches oppo base
        # 2. soldier is killed as the last agent
        # 3. soldier kills the last oppo agent
        if (not can_soldier_act and not already_killed[field.soldier_idx]) or rews_dones[1][field.soldier_idx] or (all(field.killed[field.team_red]) and soldier_died) or (all(field.killed[field.team_blue]) and soldier_rew > 0):
            transition_s = (soldier_obs, soldier_act, soldier_rew, soldier_died or rews_dones[1][field.soldier_idx] or (all(field.killed[field.team_blue]) and soldier_rew > 0), new_obs[field.soldier_idx])
            replay_buffer_s.append(transition_s)
            episode_reward += soldier_rew
            episode_reward_s += soldier_rew

        field, obs = reset_env(env, field)
        can_soldier_act = False
        soldier_act = None
        soldier_obs = None
        soldier_rew = None
        soldier_new_obs = None
        soldier_died = None
        env_reset = True

        reward_buffer.append(episode_reward)
        soldier_reward_buffer.append(episode_reward_s)
        episode_reward = 0.0
        episode_reward_s = 0.0

    ## Episode is not concluded and the soldier hasn't acted in this step
    ## Otherwise ignore the half completed action since effect is not known
    ## Put it afterwards because there would be double attribution of the transition for soldier otherwise
    ## Don't add the transition if agent is already killed and did not take an action in the previous step
    if not env_reset and not can_soldier_act and (not already_killed[field.soldier_idx] or soldier_died):
        ## This is when the effect of the action takes place so save this in replay buffer
        ## If the agent died in the very previous turn then use the reward and mark as episode done for the agent only
        transition_s = (soldier_obs, soldier_act, soldier_rew, soldier_died or False, soldier_new_obs)
        replay_buffer_s.append(transition_s)
        episode_reward += soldier_rew
        episode_reward_s += soldier_rew
        if soldier_died:
            soldier_died = None

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
        torch.save(target_net.state_dict(), f'dqn_Shoot-Multi-NonM-Tank_{TIME_STR}_s{step}.pth')
        torch.save(target_net_s.state_dict(), f'dqn_Shoot-Multi-NonM-Soldier_{TIME_STR}_s{step}.pth')
    
    if step == MAX_STEPS:
        break
