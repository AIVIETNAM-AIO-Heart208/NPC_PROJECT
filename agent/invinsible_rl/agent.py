import random
from collections import deque
from pathlib import Path

try:
    import os
    import torch
    import torch.nn as nn
except Exception:
    os = None
    torch = None
    nn = None


MODES = ("FARM_BOX", "COLLECT_ITEM", "HUNT", "TRAP", "SAFE_REPOSITION")


if nn is not None:
    class ModeQNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)


class Agent:
    """
    Hybrid rule-based agent:
    - chain-reaction aware timed danger,
    - safety override before every objective,
    - early box farming and item pickup,
    - late pressure/trap behavior when a safe bomb line exists.
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    team_id = "InvinsibleHybrid"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.mode_net = None
        self.mode_input_dim = 12
        self._load_mode_model()

    def act(self, obs):
        self.turn += 1
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        occupied = set(bomb_positions)
        occupied.discard(my_pos)

        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]

        danger_time, block_until = self._timed_danger(grid, bombs, players)
        urgent_timer = danger_time.get(my_pos)
        if urgent_timer is not None and urgent_timer <= 3:
            escape = self._escape_action(grid, my_pos, occupied, danger_time, block_until)
            if escape is not None:
                return escape
            loose = self._least_bad_move(grid, my_pos, occupied, danger_time)
            return loose if loose is not None else 0

        valid_actions = self._valid_actions(grid, my_pos, occupied)
        context = {
            "grid": grid,
            "players": players,
            "bombs": bombs,
            "my_pos": my_pos,
            "bomb_radius": bomb_radius,
            "bombs_left": int(bombs_left),
            "radius_bonus": int(bomb_bonus),
            "bomb_positions": bomb_positions,
            "occupied": occupied,
            "enemies": enemies,
            "danger_time": danger_time,
            "block_until": block_until,
            "valid_actions": valid_actions,
        }

        mode_order = self._mode_order(obs, context)
        for mode in mode_order:
            action = self._execute_mode(mode, context)
            if action is not None:
                return action

        safe_moves = [
            a for a in valid_actions
            if self._is_safe_at(self._next_pos(my_pos, a), 1, danger_time, block_until)
        ]
        if 0 in safe_moves and len(safe_moves) > 1:
            safe_moves.remove(0)
        return self._best_roam_action(grid, my_pos, safe_moves, occupied, enemies) if safe_moves else 0

    def _load_mode_model(self):
        if torch is None:
            return
        here = Path(__file__).resolve().parent
        candidates = []
        if os is not None and os.environ.get("MODE_DQN_CHECKPOINT"):
            candidates.append(Path(os.environ["MODE_DQN_CHECKPOINT"]))
        candidates.extend([here / "mode_model.pth", here / "mode_latest.pth"])
        for path in candidates:
            if not path.exists():
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu")
                input_dim = int(checkpoint.get("input_dim", self.mode_input_dim))
                self.mode_input_dim = input_dim
                self.mode_net = ModeQNet(input_dim, len(MODES))
                self.mode_net.load_state_dict(checkpoint["model_state_dict"])
                self.mode_net.eval()
                return
            except Exception:
                self.mode_net = None

    def _mode_features(self, obs, context):
        grid = context["grid"]
        players = context["players"]
        my_pos = context["my_pos"]
        enemies = context["enemies"]
        danger_time = context["danger_time"]
        bomb_radius = context["bomb_radius"]
        bombs_left = context["bombs_left"]
        radius_bonus = context["radius_bonus"]
        boxes = sum(1 for x in range(grid.shape[0]) for y in range(grid.shape[1]) if int(grid[x, y]) == 2)
        items = sum(1 for x in range(grid.shape[0]) for y in range(grid.shape[1]) if int(grid[x, y]) in (3, 4))
        nearest_item = self._nearest_distance(my_pos, self._item_tiles(grid, bombs_left, radius_bonus), default=26)
        nearest_enemy = self._nearest_distance(my_pos, set(enemies), default=26)
        current_bomb_score = self._bomb_score(grid, players, my_pos, bomb_radius, enemies)
        current_danger = danger_time.get(my_pos, 8)
        safe_open = self._open_neighbors(grid, my_pos, context["occupied"])
        return [
            min(self.turn, 500) / 500.0,
            bombs_left / 5.0,
            radius_bonus / 5.0,
            len(enemies) / 3.0,
            boxes / 80.0,
            items / 20.0,
            nearest_item / 26.0,
            nearest_enemy / 26.0,
            min(current_bomb_score, 12.0) / 12.0,
            min(current_danger, 8) / 8.0,
            safe_open / 4.0,
            1.0 if self._can_escape_after_placing(grid, my_pos, context["occupied"], context["bombs"], players, bomb_radius) else 0.0,
        ]

    def _mode_order(self, obs, context):
        if self.mode_net is not None and torch is not None:
            try:
                feat = torch.tensor([self._mode_features(obs, context)], dtype=torch.float32)
                with torch.no_grad():
                    q = self.mode_net(feat).squeeze(0).tolist()
                modes = [m for _, m in sorted(zip(q, MODES), reverse=True)]
                return modes + [m for m in MODES if m not in modes]
            except Exception:
                pass

        grid = context["grid"]
        enemies = context["enemies"]
        item_tiles = self._item_tiles(grid, context["bombs_left"], context["radius_bonus"])
        item_dist = self._nearest_distance(context["my_pos"], item_tiles, default=26)
        bomb_score = self._bomb_score(grid, context["players"], context["my_pos"], context["bomb_radius"], enemies)
        late = self.turn >= 220 or len(enemies) <= 2
        if late and bomb_score >= self._bomb_threshold():
            return ["TRAP", "COLLECT_ITEM", "FARM_BOX", "HUNT", "SAFE_REPOSITION"]
        if item_tiles and (
            context["bombs_left"] <= 1
            or context["radius_bonus"] <= 1
            or (self.turn >= 180 and item_dist <= 5)
        ):
            return ["COLLECT_ITEM", "FARM_BOX", "TRAP", "HUNT", "SAFE_REPOSITION"]
        if self.turn < 220:
            return ["FARM_BOX", "COLLECT_ITEM", "TRAP", "HUNT", "SAFE_REPOSITION"]
        return ["TRAP", "HUNT", "COLLECT_ITEM", "FARM_BOX", "SAFE_REPOSITION"]

    def _execute_mode(self, mode, context):
        grid = context["grid"]
        players = context["players"]
        bombs = context["bombs"]
        my_pos = context["my_pos"]
        occupied = context["occupied"]
        enemies = context["enemies"]
        danger_time = context["danger_time"]
        block_until = context["block_until"]
        bomb_radius = context["bomb_radius"]
        bombs_left = context["bombs_left"]

        if mode == "COLLECT_ITEM":
            targets = self._item_tiles(grid, context["bombs_left"], context["radius_bonus"])
            return self._move_to_targets(grid, my_pos, targets, occupied, danger_time, block_until, max_depth=10)

        if mode == "FARM_BOX":
            if bombs_left <= 0:
                return None
            spot = self._best_bomb_spot(grid, players, bombs, my_pos, occupied, bomb_radius, enemies, prefer_enemy=False)
            return self._action_for_bomb_spot(grid, players, bombs, my_pos, occupied, danger_time, block_until, bomb_radius, spot)

        if mode == "TRAP":
            if bombs_left <= 0:
                return None
            current_score = self._bomb_score(grid, players, my_pos, bomb_radius, enemies)
            if current_score >= self._bomb_threshold() and self._can_escape_after_placing(grid, my_pos, occupied, bombs, players, bomb_radius):
                return 5
            spot = self._best_bomb_spot(grid, players, bombs, my_pos, occupied, bomb_radius, enemies, prefer_enemy=True)
            return self._action_for_bomb_spot(grid, players, bombs, my_pos, occupied, danger_time, block_until, bomb_radius, spot)

        if mode == "HUNT":
            targets = self._enemy_pressure_tiles(grid, enemies, occupied)
            return self._move_to_targets(grid, my_pos, targets, occupied, danger_time, block_until, max_depth=12)

        if mode == "SAFE_REPOSITION":
            safe_moves = [
                a for a in context["valid_actions"]
                if self._is_safe_at(self._next_pos(my_pos, a), 1, danger_time, block_until)
            ]
            if 0 in safe_moves and len(safe_moves) > 1:
                safe_moves.remove(0)
            return self._best_roam_action(grid, my_pos, safe_moves, occupied, enemies) if safe_moves else None
        return None

    def _action_for_bomb_spot(self, grid, players, bombs, my_pos, occupied, danger_time, block_until, bomb_radius, spot):
        if spot is None:
            return None
        if spot == my_pos:
            if self._can_escape_after_placing(grid, my_pos, occupied, bombs, players, bomb_radius):
                return 5
            return None
        return self._move_to_targets(grid, my_pos, {spot}, occupied, danger_time, block_until, max_depth=10)

    def _nearest_distance(self, src, targets, default=26):
        if not targets:
            return default
        return min(abs(src[0] - x) + abs(src[1] - y) for x, y in targets)

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (0, 3, 4)

    def _valid_actions(self, grid, my_pos, occupied):
        actions = [0]
        for a in (1, 2, 3, 4):
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                actions.append(a)
        return actions

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _bomb_infos(self, grid, bombs, players, extra_bomb=None):
        infos = []
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0:
                continue
            radius = 2
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            infos.append({
                "pos": (bx, by),
                "timer": max(1, timer),
                "blast": self._blast_tiles(grid, bx, by, radius),
            })

        if extra_bomb is not None:
            bx, by = extra_bomb["pos"]
            radius = int(extra_bomb["radius"])
            infos.append({
                "pos": (bx, by),
                "timer": int(extra_bomb["timer"]),
                "blast": self._blast_tiles(grid, bx, by, radius),
            })
        return infos

    def _chain_times(self, bomb_infos):
        times = [info["timer"] for info in bomb_infos]
        changed = True
        loops = 0
        while changed and loops < len(bomb_infos) + 2:
            changed = False
            loops += 1
            for i, info in enumerate(bomb_infos):
                for j, other in enumerate(bomb_infos):
                    if i == j:
                        continue
                    if other["pos"] in info["blast"] and times[i] < times[j]:
                        times[j] = times[i]
                        changed = True
        return times

    def _timed_danger(self, grid, bombs, players, extra_bomb=None):
        infos = self._bomb_infos(grid, bombs, players, extra_bomb=extra_bomb)
        if not infos:
            return {}, {}

        times = self._chain_times(infos)
        danger_time = {}
        block_until = {}
        for info, timer in zip(infos, times):
            block_until[info["pos"]] = timer
            for tile in info["blast"]:
                old = danger_time.get(tile)
                if old is None or timer < old:
                    danger_time[tile] = timer
        return danger_time, block_until

    def _is_safe_at(self, pos, arrival_t, danger_time, block_until):
        blocked = block_until.get(pos)
        if blocked is not None and arrival_t <= blocked:
            return False
        danger = danger_time.get(pos)
        return danger is None or arrival_t < danger

    def _escape_action(self, grid, start, occupied, danger_time, block_until, max_depth=9):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -10**9

        while q:
            pos, dist, first = q.popleft()
            danger = danger_time.get(pos)
            if dist > 0 and danger is None:
                open_score = self._open_neighbors(grid, pos, occupied)
                score = 20 - dist + open_score
                if score > best_score:
                    best_score = score
                    best = first
            if dist >= max_depth:
                continue
            for a in (1, 2, 3, 4, 0):
                npos = self._next_pos(pos, a)
                nt = dist + 1
                if a != 0 and (not self._passable(grid, npos[0], npos[1]) or npos in occupied):
                    continue
                if not self._is_safe_at(npos, nt, danger_time, block_until):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, nt, a if first is None else first))

        return best

    def _least_bad_move(self, grid, my_pos, occupied, danger_time):
        best = None
        best_timer = -1
        for a in self._valid_actions(grid, my_pos, occupied):
            if a == 0:
                continue
            npos = self._next_pos(my_pos, a)
            timer = danger_time.get(npos, 99)
            if timer > best_timer:
                best_timer = timer
                best = a
        return best

    def _move_to_targets(self, grid, start, targets, occupied, danger_time, block_until, max_depth=12):
        if not targets:
            return None
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        while q:
            pos, dist, first_action = q.popleft()
            if dist > 0 and pos in targets:
                return first_action
            if dist >= max_depth:
                continue
            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                nt = dist + 1
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in occupied and npos not in targets:
                    continue
                if not self._is_safe_at(npos, nt, danger_time, block_until):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, nt, a if first_action is None else first_action))
        return None

    def _open_neighbors(self, grid, pos, occupied):
        count = 0
        for a in (1, 2, 3, 4):
            nx, ny = self._next_pos(pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                count += 1
        return count

    def _line_clear(self, grid, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if int(grid[ax, y]) in (1, 2):
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if int(grid[x, ay]) in (1, 2):
                    return False
            return True
        return False

    def _bomb_score(self, grid, players, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        boxes = sum(1 for x, y in blast if int(grid[x, y]) == 2)
        items = sum(1 for x, y in blast if int(grid[x, y]) in (3, 4))
        enemy_hits = 0
        trap_bonus = 0
        for enemy in enemies:
            if enemy in blast and self._line_clear(grid, pos, enemy):
                enemy_hits += 1
                trap_bonus += max(0, 3 - self._open_neighbors(grid, enemy, set()))

        late = self.turn >= 220 or len(enemies) <= 2
        return (
            boxes * (3.0 if not late else 1.8)
            + enemy_hits * (7.5 if late else 5.0)
            + trap_bonus * 1.5
            - items * 0.5
        )

    def _bomb_threshold(self):
        if self.turn < 180:
            return 2.8
        if self.turn < 300:
            return 2.0
        return 1.2

    def _can_escape_after_placing(self, grid, my_pos, occupied, bombs, players, bomb_radius):
        extra_bomb = {"pos": my_pos, "timer": 7, "radius": bomb_radius}
        danger_time, block_until = self._timed_danger(grid, bombs, players, extra_bomb=extra_bomb)
        return self._escape_action(grid, my_pos, occupied | {my_pos}, danger_time, block_until, max_depth=9) is not None

    def _candidate_tiles(self, grid, start, occupied, max_depth=9):
        q = deque([(start, 0)])
        seen = {start}
        out = {start}
        while q:
            pos, dist = q.popleft()
            if dist >= max_depth:
                continue
            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]) or npos in occupied:
                    continue
                seen.add(npos)
                out.add(npos)
                q.append((npos, dist + 1))
        return out

    def _best_bomb_spot(self, grid, players, bombs, my_pos, occupied, radius, enemies, prefer_enemy=False):
        danger_time, block_until = self._timed_danger(grid, bombs, players)
        best_spot = None
        best_score = self._bomb_threshold()
        candidates = self._candidate_tiles(grid, my_pos, occupied, max_depth=9)

        for pos in candidates:
            if pos in occupied:
                continue
            dist = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
            if not self._is_safe_at(pos, max(1, dist), danger_time, block_until):
                continue
            score = self._bomb_score(grid, players, pos, radius, enemies) - 0.20 * dist
            if prefer_enemy:
                if not self._bomb_hits_enemy(grid, pos, radius, enemies):
                    score -= 2.5
                score += self._trap_potential(grid, pos, radius, enemies)
            else:
                score -= self._trap_potential(grid, pos, radius, enemies) * 0.2
            if score <= best_score:
                continue
            if not self._can_escape_after_placing(grid, pos, occupied, bombs, players, radius):
                continue
            best_score = score
            best_spot = pos
        return best_spot

    def _bomb_hits_enemy(self, grid, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return any(enemy in blast and self._line_clear(grid, pos, enemy) for enemy in enemies)

    def _trap_potential(self, grid, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        score = 0.0
        for enemy in enemies:
            if enemy in blast and self._line_clear(grid, pos, enemy):
                score += max(0, 4 - self._open_neighbors(grid, enemy, set())) * 1.2
            else:
                dist = abs(pos[0] - enemy[0]) + abs(pos[1] - enemy[1])
                if dist <= radius + 1:
                    score += 0.4
        return score

    def _item_tiles(self, grid, bombs_left=1, radius_bonus=0):
        preferred = set()
        if bombs_left <= 1:
            preferred.add(4)
        if radius_bonus <= 1:
            preferred.add(3)
        tiles = {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in preferred
        }
        if tiles:
            return tiles
        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in (3, 4)
        }

    def _enemy_pressure_tiles(self, grid, enemies, occupied):
        targets = set()
        for ex, ey in enemies:
            for a in (1, 2, 3, 4):
                dx, dy = self.MOVES[a]
                for r in range(1, 4):
                    x, y = ex + dx * r, ey + dy * r
                    if not self._passable(grid, x, y):
                        break
                    if (x, y) not in occupied:
                        targets.add((x, y))
        return targets

    def _best_roam_action(self, grid, my_pos, actions, occupied, enemies):
        if not actions:
            return 0
        center = (grid.shape[0] // 2, grid.shape[1] // 2)
        best_action = actions[0]
        best_score = -10**9
        for action in actions:
            npos = self._next_pos(my_pos, action)
            if action == 0 and len(actions) > 1:
                continue
            open_score = self._open_neighbors(grid, npos, occupied)
            center_dist = abs(npos[0] - center[0]) + abs(npos[1] - center[1])
            enemy_dist = min((abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies), default=6)
            score = open_score * 2.0 - center_dist * 0.15 - max(0, 2 - enemy_dist) * 0.8
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
