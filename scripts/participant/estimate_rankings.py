import os
import argparse
import random
import sys
from pathlib import Path

import trueskill
import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


def _empty_stats(n_players=4):
    return [{"kills": 0, "boxes": 0, "items": 0, "bombs": 0} for _ in range(n_players)]


def _blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    h, w = grid.shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < h and 0 <= y < w):
                break
            cell = int(grid[x, y])
            if cell == 1:
                break
            tiles.add((x, y))
            if cell == 2:
                break
    return tiles


def _owner_of_blast_tile(obs, x, y):
    for b in obs["bombs"]:
        bx, by, _timer, owner_id = [int(v) for v in b]
        radius = 1 + int(obs["players"][owner_id][4])
        if (x, y) in _blast_tiles(obs["map"], bx, by, radius):
            return owner_id
    return None


def _update_stats(stats, prev_obs, next_obs, actions):
    prev_players = prev_obs["players"]
    next_players = next_obs["players"]
    prev_grid = prev_obs["map"]
    next_grid = next_obs["map"]
    prev_bomb_positions = {(int(b[0]), int(b[1])) for b in prev_obs["bombs"]}

    for pid, action in enumerate(actions):
        if int(prev_players[pid][2]) != 1:
            continue
        px, py = int(prev_players[pid][0]), int(prev_players[pid][1])
        if int(action) == 5 and int(prev_players[pid][3]) > 0 and (px, py) not in prev_bomb_positions:
            stats[pid]["bombs"] += 1

        nx, ny = int(next_players[pid][0]), int(next_players[pid][1])
        occupants = sum(
            1
            for p in next_players
            if int(p[2]) == 1 and int(p[0]) == nx and int(p[1]) == ny
        )
        if int(next_players[pid][2]) == 1 and int(prev_grid[nx, ny]) in (3, 4) and occupants == 1:
            stats[pid]["items"] += 1

    for x, y in np.argwhere((prev_grid == 2) & (next_grid != 2)):
        owner = _owner_of_blast_tile(prev_obs, int(x), int(y))
        if owner is not None:
            stats[owner]["boxes"] += 1

    for victim_id, prev_player in enumerate(prev_players):
        if int(prev_player[2]) != 1 or int(next_players[victim_id][2]) == 1:
            continue
        owner = _owner_of_blast_tile(prev_obs, int(prev_player[0]), int(prev_player[1]))
        if owner is not None and owner != victim_id:
            stats[owner]["kills"] += 1


def _rank_key(stats, players, pid):
    alive = 1 if int(players[pid][2]) == 1 else 0
    return (
        alive,
        int(stats[pid]["kills"]),
        int(stats[pid]["boxes"]),
        int(stats[pid]["items"]),
        int(stats[pid]["bombs"]),
    )


def _ranks_from_tiebreak(stats, players):
    keys = [_rank_key(stats, players, pid) for pid in range(len(stats))]
    return [sum(1 for other in keys if other > key) for key in keys]


def estimate_rankings(agent_path, num_matches=100, max_steps=500):
    print(f"Loading agent from {agent_path}...")
    
    # Initialize TrueSkill environment with new competition defaults
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    agent_rating = ts_env.Rating()
    baseline_rating = ts_env.Rating()

    wins = 0
    draws = 0
    total_rank = 0

    env = BomberEnv(max_steps=max_steps)

    for i in range(num_matches):
        # Player 0 is the agent, Player 1-3 are random baselines
        agent_paths = [agent_path, "Random", "Random", "Random"]
        try:
            agents, names = make_agents(agent_paths, seed=None)
        except Exception as e:
            print(f"Failed to load agent: {e}")
            return
            
        agent_name = names[0]
        
        obs = env.reset()
        done = False
        step = 0
        
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_order = []
        stats = _empty_stats(4)
        
        while not done and step < max_steps:
            actions = []
            for j in range(4):
                try:
                    action = agents[j].act(obs)
                except Exception:
                    action = 0
                actions.append(action)
                
            prev_obs = obs
            obs, terminated, truncated = env.step(actions)
            _update_stats(stats, prev_obs, obs, actions)
            done = terminated or truncated
            step += 1
            
            alive_now = [bool(p[2]) for p in obs["players"]]
            for j in range(4):
                if prev_alive[j] and not alive_now[j]:
                    death_order.append(j)
            prev_alive = alive_now
            
        ranks = _ranks_from_tiebreak(stats, obs["players"])

        if ranks[0] == 0 and all(r > 0 for r in ranks[1:]):
            wins += 1
        elif ranks[0] == 0:
            draws += 1
            
        total_rank += ranks[0]
        
        # Update TrueSkill
        rating_groups = [(agent_rating,), (baseline_rating,), (baseline_rating,), (baseline_rating,)]
        new_ratings = ts_env.rate(rating_groups, ranks=ranks)
        agent_rating = new_ratings[0][0]
        
        # Print progress
        score = agent_rating.mu - 3 * agent_rating.sigma
        print(f"Match {i+1}/{num_matches} | Rank: {ranks[0]} | Est. Score: {score:.2f} | mu: {agent_rating.mu:.2f}, sigma: {agent_rating.sigma:.2f}", end="\r")

    print("\n\n=== Final Estimated Results ===")
    print(f"Agent: {agent_name}")
    print(f"Matches Played: {num_matches}")
    print(f"Win Rate: {(wins / num_matches) * 100:.1f}%")
    print(f"Draw Rate: {(draws / num_matches) * 100:.1f}%")
    print(f"Average Rank: {total_rank / num_matches:.2f} (0 is winner, 3 is first to die)")
    print(f"Estimated TrueSkill: Score = {agent_rating.mu - 3 * agent_rating.sigma:.2f} (mu={agent_rating.mu:.2f}, sigma={agent_rating.sigma:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate agent rankings by playing against random baselines.")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to your agent.py file or agent folder.")
    parser.add_argument("--num_matches", type=int, default=100, help="Number of matches to simulate.")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per match.")
    args = parser.parse_args()
    
    estimate_rankings(args.agent_path, args.num_matches, args.max_steps)
