import random
from collections import deque


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

        if int(bombs_left) > 0 and my_pos not in bomb_positions:
            current_score = self._bomb_score(grid, players, my_pos, bomb_radius, enemies)
            if current_score >= self._bomb_threshold() and self._can_escape_after_placing(
                grid, my_pos, occupied, bombs, players, bomb_radius
            ):
                return 5

        item_targets = self._item_tiles(grid, bombs_left=int(bombs_left), radius_bonus=int(bomb_bonus))
        if item_targets:
            move = self._move_to_targets(grid, my_pos, item_targets, occupied, danger_time, block_until, max_depth=10)
            if move is not None:
                return move

        if int(bombs_left) > 0:
            spot = self._best_bomb_spot(grid, players, bombs, my_pos, occupied, bomb_radius, enemies)
            if spot is not None:
                if spot == my_pos and self._can_escape_after_placing(grid, my_pos, occupied, bombs, players, bomb_radius):
                    return 5
                move = self._move_to_targets(grid, my_pos, {spot}, occupied, danger_time, block_until, max_depth=10)
                if move is not None:
                    return move

        if enemies:
            pressure_tiles = self._enemy_pressure_tiles(grid, enemies, occupied)
            move = self._move_to_targets(grid, my_pos, pressure_tiles, occupied, danger_time, block_until, max_depth=12)
            if move is not None:
                return move

        safe_moves = [
            a for a in valid_actions
            if self._is_safe_at(self._next_pos(my_pos, a), 1, danger_time, block_until)
        ]
        if 0 in safe_moves and len(safe_moves) > 1:
            safe_moves.remove(0)
        return self._best_roam_action(grid, my_pos, safe_moves, occupied, enemies) if safe_moves else 0

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

    def _best_bomb_spot(self, grid, players, bombs, my_pos, occupied, radius, enemies):
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
            if score <= best_score:
                continue
            if not self._can_escape_after_placing(grid, pos, occupied, bombs, players, radius):
                continue
            best_score = score
            best_spot = pos
        return best_spot

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
