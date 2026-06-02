from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim


# Constants from engine
class Map:
    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4
    BOMB = 5

class Player:
    MAX_BOMB_RADIUS = 5
    MAX_BOMB_CAPACITY = 5

BOMB_MAX_TIMER = 7

class ReplayBuffer:
    """Pre-allocated numpy circular buffer — sample() is pure array indexing, no Python objects."""
    def __init__(self, capacity: int, map_shape, aux_dim: int):
        self.capacity  = capacity
        self.pos       = 0
        self.size      = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim   = int(aux_dim)
        self.map_states      = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states      = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int64)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)
        self.next_action_masks = np.ones((capacity, 6),     dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done, next_action_mask=None):
        self.map_states[self.pos]      = map_state
        self.aux_states[self.pos]      = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done
        if next_action_mask is None:
            self.next_action_masks[self.pos] = 1.0
        else:
            self.next_action_masks[self.pos] = next_action_mask
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx],
            self.aux_states[idx],
            self.next_map_states[idx],
            self.next_aux_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_action_masks[idx],
        )

class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    """
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)

def encode_obs(obs, agent_ids, output_channels=12):
    """
    Returns:
      map_feat: spatial tensor for Conv2D branch, shape (C, H, W)
      aux_feat: scalar tensor for auxiliary branch, shape (A,)

    agent_ids: list/tuple whose first item is the controlled player id.
    output_channels=12 is the fixed 4-player encoder used for new training.
    output_channels=9 keeps compatibility with the old bundled checkpoint.
    """
    if obs is None:
        raise ValueError("obs should not be None")

    user_id = int(agent_ids[0])

    grid    = obs["map"]      # (H, W)
    players = obs["players"]  # (num_players, 5)
    bombs   = obs["bombs"]    # (N, 4), N may be 0
    H, W    = grid.shape

    # One-hot map: grass, wall, box, item_radius, item_capacity
    map_channels = []
    for v in [Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        map_channels.append((grid == v).astype(np.float32))
    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    my_pos  = np.zeros((H, W), dtype=np.float32)
    enemy_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive)  == 1:
        my_pos[int(my_x), int(my_y)] = 1.0

    enemies_alive = 0
    nearest_enemy_dist = H + W
    for pid, p in enumerate(players):
        if pid == user_id or int(p[2]) != 1:
            continue
        enemies_alive += 1
        ex, ey = int(p[0]), int(p[1])
        enemy_pos[ex, ey] = 1.0
        nearest_enemy_dist = min(nearest_enemy_dist, abs(int(my_x) - ex) + abs(int(my_y) - ey))

    # Bomb channels — bombs is a numpy array, not a list of Bomb objects
    effective_timers = _effective_bomb_timers_np(grid, players, bombs)
    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    enemy_bomb = np.zeros((H, W), dtype=np.float32)
    for idx, b in enumerate(bombs):
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        t = float(effective_timers.get(idx, int(timer))) / BOMB_MAX_TIMER
        bomb_timer[bx, by] = max(bomb_timer[bx, by], t)
        if int(owner_id) == user_id:
            bomb_owned[bx, by] = 1.0
        else:
            enemy_bomb[bx, by] = 1.0

    scalar = np.array([
        float(my_bombs_left)   / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        float(enemies_alive) / max(1.0, float(len(players) - 1)),
        float(nearest_enemy_dist) / float(H + W),
    ], dtype=np.float32)

    if output_channels == 9:
        map_feat = np.stack([
            *map_channels,
            my_pos,
            enemy_pos,
            bomb_timer,
            bomb_owned,
        ], axis=0).astype(np.float32)
        return map_feat, scalar[:3]

    map_feat = np.stack([
        *map_channels,          # 5 channels
        my_pos,                 # 1 channel
        enemy_pos,              # 1 channel
        bomb_timer,             # 1 channel
        bomb_owned,             # 1 channel
        enemy_bomb,             # 1 channel
        _danger_channel(grid, players, bombs),  # 1 channel
        np.zeros((H, W), dtype=np.float32),     # reserved channel
    ], axis=0).astype(np.float32)  # (12, H, W)
    return map_feat, scalar


def _danger_channel(grid, players, bombs):
    danger = np.zeros_like(grid, dtype=np.float32)
    effective_timers = _effective_bomb_timers_np(grid, players, bombs)
    for idx, b in enumerate(bombs):
        bx, by, timer, owner_id = [int(v) for v in b]
        timer = effective_timers.get(idx, timer)
        if timer > 3:
            continue
        radius = 1
        if 0 <= owner_id < len(players):
            radius = 1 + int(players[owner_id][4])
        for x, y in _blast_tiles_np(grid, bx, by, radius):
            danger[x, y] = max(danger[x, y], (4.0 - float(timer)) / 3.0)
    return danger


def _effective_bomb_timers_np(grid, players, bombs):
    bombs_arr = list(bombs)
    if not bombs_arr:
        return {}

    timers = {}
    blasts = {}
    for idx, b in enumerate(bombs_arr):
        bx, by, timer, owner_id = [int(v) for v in b]
        radius = 1
        if 0 <= owner_id < len(players):
            radius = 1 + int(players[owner_id][4])
        timers[idx] = int(timer)
        blasts[idx] = set(_blast_tiles_np(grid, bx, by, radius))

    changed = True
    while changed:
        changed = False
        for trigger_idx, trigger_tiles in blasts.items():
            trigger_timer = timers[trigger_idx]
            for target_idx, b in enumerate(bombs_arr):
                if target_idx == trigger_idx:
                    continue
                bx, by = int(b[0]), int(b[1])
                if (bx, by) in trigger_tiles and timers[target_idx] > trigger_timer:
                    timers[target_idx] = trigger_timer
                    changed = True
    return timers


def _blast_tiles_np(grid, bx, by, radius):
    tiles = [(bx, by)]
    h, w = grid.shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < h and 0 <= y < w):
                break
            cell = int(grid[x, y])
            if cell == Map.WALL:
                break
            tiles.append((x, y))
            if cell == Map.BOX:
                break
    return tiles


def legal_actions(obs, agent_id):
    grid = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    if int(players[agent_id][2]) != 1:
        return [0]

    x, y = int(players[agent_id][0]), int(players[agent_id][1])
    bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
    actions = [0]

    for action, (dx, dy) in {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }.items():
        nx, ny = x + dx, y + dy
        if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
            continue
        if int(grid[nx, ny]) in (Map.WALL, Map.BOX):
            continue
        if (nx, ny) in bomb_positions:
            continue
        actions.append(action)

    if int(players[agent_id][3]) > 0 and (x, y) not in bomb_positions:
        actions.append(5)
    return actions


def action_mask(valid_actions, num_actions=6):
    mask = np.zeros(num_actions, dtype=np.float32)
    for action in valid_actions:
        if 0 <= int(action) < num_actions:
            mask[int(action)] = 1.0
    if mask.sum() == 0:
        mask[0] = 1.0
    return mask


def _empty_episode_stats(num_players=4):
    return [
        {"kills": 0, "boxes": 0, "items": 0, "bombs": 0}
        for _ in range(num_players)
    ]


def _update_episode_stats(stats, prev_obs, next_obs, actions):
    prev_players = prev_obs["players"]
    next_players = next_obs["players"]
    prev_grid = prev_obs["map"]
    next_grid = next_obs["map"]
    prev_bombs = prev_obs["bombs"]
    prev_bomb_positions = {(int(b[0]), int(b[1])) for b in prev_bombs}

    for pid, action in enumerate(actions):
        if int(prev_players[pid][2]) != 1:
            continue
        px, py = int(prev_players[pid][0]), int(prev_players[pid][1])
        if (
            int(action) == 5
            and int(prev_players[pid][3]) > 0
            and (px, py) not in prev_bomb_positions
        ):
            stats[pid]["bombs"] += 1

        nx, ny = int(next_players[pid][0]), int(next_players[pid][1])
        occupants = sum(
            1
            for p in next_players
            if int(p[2]) == 1 and int(p[0]) == nx and int(p[1]) == ny
        )
        if (
            int(next_players[pid][2]) == 1
            and int(prev_grid[nx, ny]) in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY)
            and occupants == 1
        ):
            stats[pid]["items"] += 1

    for x, y in np.argwhere((prev_grid == Map.BOX) & (next_grid != Map.BOX)):
        owner = _owner_of_blast_tile(prev_obs, int(x), int(y))
        if owner is not None and 0 <= owner < len(stats):
            stats[owner]["boxes"] += 1

    for victim_id, prev_player in enumerate(prev_players):
        if int(prev_player[2]) != 1 or int(next_players[victim_id][2]) == 1:
            continue
        vx, vy = int(prev_player[0]), int(prev_player[1])
        owner = _owner_of_blast_tile(prev_obs, vx, vy)
        if owner is not None and owner != victim_id and 0 <= owner < len(stats):
            stats[owner]["kills"] += 1


def _owner_of_blast_tile(obs, x, y):
    grid = obs["map"]
    players = obs["players"]
    for b in obs["bombs"]:
        bx, by, _timer, owner_id = [int(v) for v in b]
        radius = 1
        if 0 <= owner_id < len(players):
            radius = 1 + int(players[owner_id][4])
        if (x, y) in _blast_tiles_np(grid, bx, by, radius):
            return owner_id
    return None


def _ranking_key_from_stats(stats, players, pid):
    alive = 1 if int(players[pid][2]) == 1 else 0
    return (
        alive,
        int(stats[pid]["kills"]),
        int(stats[pid]["boxes"]),
        int(stats[pid]["items"]),
        int(stats[pid]["bombs"]),
    )


def _tiebreak_rank(stats, players, agent_id):
    keys = [_ranking_key_from_stats(stats, players, pid) for pid in range(len(stats))]
    my_key = keys[agent_id]
    return sum(1 for key in keys if key > my_key)


def _terminal_tiebreak_bonus(stats, players, agent_id):
    rank = _tiebreak_rank(stats, players, agent_id)
    bonuses = [8.0, 3.0, -2.0, -5.0]
    return bonuses[min(rank, len(bonuses) - 1)]

class TrainingAgent:
    """
    Agent class for DQN training and evaluation.
    Args:
        agent_id: int
        input_dim: int
        num_actions: int
        lr: float
        device: str
        pretrained_model: str
    Returns:
        None
    """
    team_id = "DQNAgent"
    
    def __init__(self, agent_id: int, input_spec, num_actions: int, lr: float=1e-3, device: str="cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.epsilon = 1.0

        # Networks: Q-Network (learning) and Target-Network (stable target)
        if pretrained_model:
            self.load_agent(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)

        self.target_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync weights initially
        
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, map_state, aux_state, epsilon=0.0, valid_actions=None):
        """
        Take an action based on the state.
        Args:
            map_state: np.ndarray
            aux_state: np.ndarray
            epsilon: float
        Returns:
            action: int
        """
        valid_actions = list(range(self.num_actions)) if valid_actions is None else list(valid_actions)
        if not valid_actions:
            return 0

        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(map_tensor, aux_tensor).squeeze(0)
            mask = torch.full_like(q_values, -1e9)
            mask[torch.tensor(valid_actions, dtype=torch.long, device=self.device)] = 0.0
            action = (q_values + mask).argmax().item()
            
        # action with the highest predicted Q-value
        return action

    def train_step(self, map_state, aux_state, next_map_state, next_aux_state, action, reward, done, next_action_mask):
        """
        Train the DQN agent for one step.
        Args:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
        Returns:
            None
        """
        # torch.from_numpy is zero-copy; only move to device when not CPU
        map_state_t      = torch.from_numpy(map_state)
        aux_state_t      = torch.from_numpy(aux_state)
        next_map_state_t = torch.from_numpy(next_map_state)
        next_aux_state_t = torch.from_numpy(next_aux_state)
        action_t     = torch.from_numpy(action).unsqueeze(1)
        reward_t     = torch.from_numpy(reward).unsqueeze(1)
        done_t       = torch.from_numpy(done).unsqueeze(1)
        next_action_mask_t = torch.from_numpy(next_action_mask)
        if self.device != "cpu":
            map_state_t      = map_state_t.to(self.device)
            aux_state_t      = aux_state_t.to(self.device)
            next_map_state_t = next_map_state_t.to(self.device)
            next_aux_state_t = next_aux_state_t.to(self.device)
            action_t     = action_t.to(self.device)
            reward_t     = reward_t.to(self.device)
            done_t       = done_t.to(self.device)
            next_action_mask_t = next_action_mask_t.to(self.device)

        # 2. Calculate current Q-values: Q(s, a)
        # gather() extracts the Q-value for the specific action taken
        q_values = self.q_net(map_state_t, aux_state_t).gather(1, action_t)

        with torch.no_grad():
            # Double DQN: online net chooses the next action, target net prices it.
            next_online_q = self.q_net(next_map_state_t, next_aux_state_t)
            next_online_q = next_online_q.masked_fill(next_action_mask_t <= 0.0, -1e9)
            next_action_t = next_online_q.argmax(1, keepdim=True)

            next_target_q = self.target_net(next_map_state_t, next_aux_state_t)
            max_next_q = next_target_q.gather(1, next_action_t)
            target_q   = reward_t + self.gamma * max_next_q * (1 - done_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad(set_to_none=True)  # skip memset, just nullify refs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.global_step += 1
        return loss.item()
        
    def update_target_network(self):
        """Copies the learned weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def load_agent(self, pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=self.device)
        input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint["input_dim"]))
        self.map_shape = tuple(input_spec[0])
        self.aux_dim = int(input_spec[1])
        self.num_actions = checkpoint["num_actions"]
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epsilon = checkpoint["epsilon"]

def train_dqn(
    user_id=0,
    enemy_type="simple",
    num_episodes=100,
    max_steps=500,
    seed=86,
    save_model=True,
    pretrained_model=None,
    epsilon_start=None,
    epsilon_min=0.03,
    epsilon_decay=0.997,
    batch_size=128,
    buffer_capacity=100_000,
    target_update_steps=1_000,
    save_every_episodes=100,
):
    # Training-only imports - placed here so they don't run when the evaluator loads this file
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from reward import compute_reward  
    from utils import (plot_loss, plot_rewards, plot_win_rates, 
                       plot_moving_average, seed_everything, save_model_fn)
    from agent import (SimpleRuleAgent, SmarterRuleAgent, 
                       TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent)
    from engine import BomberEnv 

    env = BomberEnv(max_steps=max_steps, seed=seed)

    def make_enemy(agent_id, kind=None):
        kind = enemy_type if kind is None else kind
        if kind == "mixed":
            kind = random.choice(["simple", "smarter", "tactical", "genius", "box_farmer"])
        if kind == "simple":
            return SimpleRuleAgent(agent_id)
        if kind == "smarter":
            return SmarterRuleAgent(agent_id)
        if kind == "tactical":
            return TacticalRuleAgent(agent_id)
        if kind == "genius":
            return GeniusRuleAgent(agent_id)
        if kind == "box_farmer":
            return BoxFarmerAgent(agent_id)
        raise ValueError(f"Invalid enemy type: {kind}")

    def make_enemy_agents():
        return [make_enemy(i) for i in range(4) if i != user_id]

    enemy_agents = make_enemy_agents()

    # hyperparam
    batch_size         = int(batch_size)
    lr                 = 1e-3

    dummy_obs = env.reset(seed=seed)
    agent_ids = [user_id]
    sample_state = encode_obs(dummy_obs, agent_ids=agent_ids, output_channels=12)
    input_spec = (sample_state[0].shape, sample_state[1].shape[0])
    num_actions = 6

    user_agent = TrainingAgent(user_id, input_spec, num_actions, lr=lr, device="cuda" if torch.cuda.is_available() else "cpu", pretrained_model=pretrained_model)
    if epsilon_start is None:
        epsilon = float(user_agent.epsilon) if pretrained_model else 0.40
    else:
        epsilon = float(epsilon_start)
    input_spec = (user_agent.map_shape, user_agent.aux_dim)
    encode_channels = int(user_agent.map_shape[0])
    buffer = ReplayBuffer(capacity=int(buffer_capacity), map_shape=input_spec[0], aux_dim=input_spec[1])

    global_step = 0
    loss_history = []
    reward_history = []
    win_history = []
    model_folder = f"ckpts/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for ep in range(num_episodes):
            if enemy_type == "mixed":
                enemy_agents = make_enemy_agents()
            obs = env.reset(seed=seed + ep)
            done = False
            total_reward = 0
            episode_stats = _empty_episode_stats(num_players=len(obs["players"]))

            map_state, aux_state = encode_obs(obs, agent_ids, output_channels=encode_channels)

            for _ in range(max_steps):
                # 1. Action
                user_action = user_agent.act(
                    map_state,
                    aux_state,
                    epsilon=epsilon,
                    valid_actions=legal_actions(obs, user_id),
                )
                actions = [0, 0, 0, 0]
                actions[user_id] = user_action
                for enemy_agent in enemy_agents:
                    actions[enemy_agent.agent_id] = enemy_agent.act(obs)

                # 2. Environment Step
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                _update_episode_stats(episode_stats, obs, next_obs, actions)

                # 3. Reward
                r = compute_reward(obs, next_obs, agent_id=user_id)
                if done:
                    r += _terminal_tiebreak_bonus(episode_stats, next_obs["players"], user_id)
                total_reward += r
                reward_history.append(r)
                if done:
                    win_history.append(1 if _tiebreak_rank(episode_stats, next_obs["players"], user_id) == 0 else 0)
                
                # 4. Buffer Push
                next_map_state, next_aux_state = encode_obs(
                    next_obs,
                    agent_ids,
                    output_channels=encode_channels,
                )
                next_valid_mask = action_mask(legal_actions(next_obs, user_id), num_actions)
                buffer.push(
                    map_state,
                    aux_state,
                    user_action,
                    r,
                    next_map_state,
                    next_aux_state,
                    done,
                    next_valid_mask,
                )

                # 5. Train
                global_step += 1
                if len(buffer) >= batch_size:
                    sampled_map_state, sampled_aux_state, sampled_next_map_state, sampled_next_aux_state, sampled_action, sampled_reward, sampled_done, sampled_next_action_mask = buffer.sample(batch_size)
                    loss = user_agent.train_step(
                        sampled_map_state,
                        sampled_aux_state,
                        sampled_next_map_state,
                        sampled_next_aux_state,
                        sampled_action,
                        sampled_reward,
                        sampled_done,
                        sampled_next_action_mask,
                    )
                    loss_history.append(loss)
                    if int(target_update_steps) > 0 and user_agent.global_step % int(target_update_steps) == 0:
                        user_agent.update_target_network()

                # 6. Update
                obs       = next_obs
                map_state = next_map_state
                aux_state = next_aux_state

                # 7. Done
                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            user_agent.epsilon = epsilon
            if (
                save_model
                and save_every_episodes
                and (ep + 1) % int(save_every_episodes) == 0
            ):
                save_model_fn(
                    user_agent.q_net,
                    user_agent.optimizer,
                    user_agent.global_step,
                    user_agent.epsilon,
                    user_agent.lr,
                    input_spec,
                    num_actions,
                    f"{model_folder}/{user_agent.global_step}_global_step_ep{ep + 1}.pth",
                )
            pbar.update(1)
            pbar.set_postfix(reward=f"{total_reward:.2f}", epsilon=f"{epsilon:.3f}")

    if save_model:
        model_path = f"{model_folder}/{user_agent.global_step}_global_step.pth"
        submission_model_path = Path(__file__).resolve().parent / "model.pth"
        save_model_fn(user_agent.q_net, 
                    user_agent.optimizer, 
                    user_agent.global_step, 
                    user_agent.epsilon, 
                    user_agent.lr, 
                    input_spec,
                    num_actions,
                    model_path)
        save_model_fn(user_agent.q_net,
                    user_agent.optimizer,
                    user_agent.global_step,
                    user_agent.epsilon,
                    user_agent.lr,
                    input_spec,
                    num_actions,
                    str(submission_model_path))
        
    plot_loss(loss_history=loss_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_loss.png")
    plot_rewards(reward_history=reward_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_rewards.png")
    plot_win_rates(win_history=win_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_win_rates.png")
    plot_moving_average(data=reward_history, window_size=10, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_moving_average.png")

def training():
    from utils import seed_everything
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_type", type=str, default="simple", choices=["simple", "smarter", "tactical", "genius", "box_farmer", "mixed"])
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=86, help="Random seed for reproducibility")
    parser.add_argument("--save_model", action="store_true", help="Save model")
    parser.add_argument("--load_model", type=str, default=None, help="Load model")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--epsilon_start", type=float, default=None, help="Initial epsilon for exploration. Defaults to checkpoint epsilon when --load_model is used.")
    parser.add_argument("--epsilon_min", type=float, default=0.03, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.997, help="Episode-level epsilon decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Replay batch size")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--target_update_steps", type=int, default=1000, help="Train steps between target-network syncs")
    parser.add_argument("--save_every_episodes", type=int, default=100, help="Save intermediate checkpoints every N episodes; 0 disables")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    print("Skip training? ", args.skip_training)
    if not args.skip_training:
        train_dqn(enemy_type=args.enemy_type, 
                    num_episodes=args.num_episodes, 
                    max_steps=args.max_steps, 
                    seed=args.seed, 
                    save_model=args.save_model,
                    pretrained_model=args.load_model,
                    epsilon_start=args.epsilon_start,
                    epsilon_min=args.epsilon_min,
                    epsilon_decay=args.epsilon_decay,
                    batch_size=args.batch_size,
                    buffer_capacity=args.buffer_capacity,
                    target_update_steps=args.target_update_steps,
                    save_every_episodes=args.save_every_episodes)
    
# Mandatory for submission
class Agent:
    """DQN Agent for submission."""    
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cpu")  # Use CPU for compatibility
        self.q_net = None
        self.map_shape = ((9, 13, 13))
        self.aux_dim = 3
        self.num_actions = 6
        
        checkpoint_path = self._resolve_checkpoint_path()
        self._load_checkpoint(str(checkpoint_path))

    def _resolve_checkpoint_path(self):
        here = Path(__file__).resolve().parent
        repo_root = here.parent.parent

        env_path = os.environ.get("DQN_CHECKPOINT")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend([
            here / "model.pth",
            here / "latest_global_step.pth",
        ])

        candidates.extend(sorted(
            here.glob("*_global_step.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ))

        ckpt_root = repo_root / "ckpts"
        if ckpt_root.exists():
            candidates.extend(sorted(
                ckpt_root.glob("**/*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ))

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "No DQN checkpoint found. Train with --save_model first, or set DQN_CHECKPOINT to a .pth file."
        )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get input spec from checkpoint
            input_spec = checkpoint.get("input_spec", 
                                       checkpoint.get("input_shape", 
                                                     checkpoint["input_dim"]))
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.num_actions = checkpoint["num_actions"]
            
            # Create and load model
            self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions)
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.q_net.to(self.device)
            self.q_net.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            raise
    
    def act(self, obs):
        """
        Take an action based on observation.
        
        Args:
            obs: dict with keys 'map', 'players', 'bombs'
        
        Returns:
            action: int in range [0, 5]
        """
        try:
            # Encode observation
            map_state, aux_state = encode_obs(
                obs,
                [self.agent_id],
                output_channels=int(self.map_shape[0]),
            )
            
            # Convert to tensors and add batch dimension
            map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
            aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
            
            # Get Q-values and select best action
            with torch.no_grad():
                q_values = self.q_net(map_tensor, aux_tensor).squeeze(0)
                valid = legal_actions(obs, self.agent_id)
                mask = torch.full_like(q_values, -1e9)
                mask[torch.tensor(valid, dtype=torch.long, device=self.device)] = 0.0
                action = (q_values + mask).argmax().item()
            
            return action
        except Exception as e:
            print(f"[ERROR] Agent.act() failed: {e}")
            # Fallback to random action on error
            return 0
        

if __name__ == "__main__":
    training()