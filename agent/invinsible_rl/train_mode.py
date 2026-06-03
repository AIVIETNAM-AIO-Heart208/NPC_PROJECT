from pathlib import Path
import argparse
import importlib.util
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AGENT_DIR = Path(__file__).resolve().parent

from engine import BomberEnv
from agent import SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent
from scripts.participant.estimate_rankings import _empty_stats, _update_stats, _ranks_from_tiebreak


def _load_local_agent_module():
    spec = importlib.util.spec_from_file_location("invinsible_rl_agent", AGENT_DIR / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_local_agent = _load_local_agent_module()
InvinsibleExecutor = _local_agent.Agent
MODES = _local_agent.MODES
ModeQNet = _local_agent.ModeQNet


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class ModeLearner(InvinsibleExecutor):
    def __init__(self, agent_id, device="cpu", lr=1e-3, load_model=None):
        super().__init__(agent_id)
        self.device = torch.device(device)
        self.mode_input_dim = 12
        self.mode_net = ModeQNet(self.mode_input_dim, len(MODES)).to(self.device)
        self.target_net = ModeQNet(self.mode_input_dim, len(MODES)).to(self.device)
        self.optimizer = optim.Adam(self.mode_net.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = 0.98
        self.global_step = 0
        self.epsilon = 0.20
        if load_model:
            self.load(load_model)
        self.target_net.load_state_dict(self.mode_net.state_dict())
        self.mode_net.eval()

    def context(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        occupied = set(bomb_positions)
        occupied.discard(my_pos)
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]
        danger_time, block_until = self._timed_danger(grid, bombs, players)
        return {
            "grid": grid,
            "players": players,
            "bombs": bombs,
            "my_pos": my_pos,
            "bomb_radius": max(1, int(bomb_bonus) + 1),
            "bombs_left": int(bombs_left),
            "radius_bonus": int(bomb_bonus),
            "bomb_positions": bomb_positions,
            "occupied": occupied,
            "enemies": enemies,
            "danger_time": danger_time,
            "block_until": block_until,
            "valid_actions": self._valid_actions(grid, my_pos, occupied),
        }

    def features(self, obs):
        return np.asarray(self._mode_features(obs, self.context(obs)), dtype=np.float32)

    def choose_mode(self, obs, epsilon):
        ctx = self.context(obs)
        px, py = ctx["my_pos"]
        danger_timer = ctx["danger_time"].get((px, py))
        if danger_timer is not None and danger_timer <= 3:
            return MODES.index("SAFE_REPOSITION")
        if random.random() < epsilon:
            return random.randrange(len(MODES))
        state = torch.tensor([self._mode_features(obs, ctx)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return int(self.mode_net(state).argmax(1).item())

    def act_with_mode(self, obs, mode_idx):
        self.turn += 1
        ctx = self.context(obs)
        px, py = ctx["my_pos"]
        danger_timer = ctx["danger_time"].get((px, py))
        if danger_timer is not None and danger_timer <= 3:
            escape = self._escape_action(ctx["grid"], ctx["my_pos"], ctx["occupied"], ctx["danger_time"], ctx["block_until"])
            return escape if escape is not None else 0
        mode_order = [MODES[mode_idx]] + [m for m in MODES if m != MODES[mode_idx]]
        for mode in mode_order:
            action = self._execute_mode(mode, ctx)
            if action is not None:
                return action
        return 0

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.mode_net(states).gather(1, actions)
        with torch.no_grad():
            next_action = self.mode_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_action)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mode_net.parameters(), 10.0)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.mode_net.state_dict())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.mode_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epsilon": self.epsilon,
            "input_dim": self.mode_input_dim,
            "modes": MODES,
        }, path)
        print(f"Mode model saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.mode_input_dim = int(ckpt.get("input_dim", 12))
        self.mode_net = ModeQNet(self.mode_input_dim, len(MODES)).to(self.device)
        self.target_net = ModeQNet(self.mode_input_dim, len(MODES)).to(self.device)
        self.mode_net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer = optim.Adam(self.mode_net.parameters(), lr=1e-3, weight_decay=1e-5)
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))


def make_enemy(agent_id, enemy_type):
    if enemy_type == "weighted_mixed":
        enemy_type = random.choices(
            ["simple", "smarter", "tactical", "genius", "box_farmer"],
            weights=[1, 2, 5, 5, 2],
            k=1,
        )[0]
    if enemy_type == "simple":
        return SimpleRuleAgent(agent_id)
    if enemy_type == "smarter":
        return SmarterRuleAgent(agent_id)
    if enemy_type == "tactical":
        return TacticalRuleAgent(agent_id)
    if enemy_type == "genius":
        return GeniusRuleAgent(agent_id)
    if enemy_type == "box_farmer":
        return BoxFarmerAgent(agent_id)
    raise ValueError(enemy_type)


def mode_reward(prev_obs, next_obs, prev_stats, next_stats, agent_id, done):
    p0 = prev_stats[agent_id]
    p1 = next_stats[agent_id]
    r = 0.0
    r += (p1["kills"] - p0["kills"]) * 8.0
    r += (p1["boxes"] - p0["boxes"]) * 1.0
    r += (p1["items"] - p0["items"]) * 0.6
    r += (p1["bombs"] - p0["bombs"]) * 0.04
    if int(prev_obs["players"][agent_id][2]) == 1 and int(next_obs["players"][agent_id][2]) == 0:
        r -= 8.0
    if done:
        ranks = _ranks_from_tiebreak(next_stats, next_obs["players"])
        r += [8.0, 2.0, -2.0, -5.0][min(ranks[agent_id], 3)]
    return r


def train_mode_dqn(
    enemy_type="weighted_mixed",
    num_episodes=500,
    max_steps=500,
    seed=86,
    epsilon_start=0.20,
    epsilon_min=0.03,
    epsilon_decay=0.996,
    save_model=True,
    load_model=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = BomberEnv(max_steps=max_steps, seed=seed)
    learner = ModeLearner(0, device=device, load_model=load_model)
    epsilon = float(epsilon_start if epsilon_start is not None else learner.epsilon)
    buffer = ReplayBuffer(50_000, learner.mode_input_dim)
    batch_size = 128
    losses = []

    with tqdm(total=num_episodes, desc="Training mode DDQN") as pbar:
        for ep in range(num_episodes):
            enemies = [make_enemy(i, enemy_type) for i in range(1, 4)]
            obs = env.reset(seed=seed + ep)
            learner.turn = 0
            stats = _empty_stats(4)
            done = False
            total_reward = 0.0
            while not done:
                state = learner.features(obs)
                mode_idx = learner.choose_mode(obs, epsilon)
                action = learner.act_with_mode(obs, mode_idx)
                actions = [action] + [enemy.act(obs) for enemy in enemies]
                prev_obs = obs
                prev_stats = [dict(s) for s in stats]
                obs, terminated, truncated = env.step(actions)
                _update_stats(stats, prev_obs, obs, actions)
                done = terminated or truncated
                reward = mode_reward(prev_obs, obs, prev_stats, stats, 0, done)
                total_reward += reward
                buffer.push(state, mode_idx, reward, learner.features(obs), done)
                if len(buffer) >= batch_size:
                    losses.append(learner.train_step(buffer.sample(batch_size)))
                    if learner.global_step % 1000 == 0:
                        learner.update_target()
                if done:
                    break
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            learner.epsilon = epsilon
            pbar.update(1)
            pbar.set_postfix(reward=f"{total_reward:.2f}", epsilon=f"{epsilon:.3f}")

    if save_model:
        learner.save(AGENT_DIR / "mode_model.pth")
        learner.save(ROOT / "ckpts" / f"mode_dqn_{enemy_type}_{num_episodes}_episodes.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_type", default="weighted_mixed", choices=["simple", "smarter", "tactical", "genius", "box_farmer", "weighted_mixed"])
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--epsilon_start", type=float, default=0.20)
    parser.add_argument("--epsilon_min", type=float, default=0.03)
    parser.add_argument("--epsilon_decay", type=float, default=0.996)
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--no_save_model", action="store_true")
    args = parser.parse_args()
    train_mode_dqn(
        enemy_type=args.enemy_type,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        save_model=not args.no_save_model,
        load_model=args.load_model,
    )


if __name__ == "__main__":
    main()
