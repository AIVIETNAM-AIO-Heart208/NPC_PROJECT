"""
Optimized rule-based agent for step-500: kills > boxes > items > bombs
- Chain-aware danger detection
- Time-aware pathfinding 
- Proper escape validation
- Smart bomb placement
"""

import random
from collections import deque


class OptimizedRuleAgent:
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    
    team_id = "OptimizedRuleAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.escape_mode = False

    def act(self, obs):
        self.turn += 1
        
        grid = obs["map"]
        players = obs["players"]
        bombs_obs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bombs_left = int(bombs_left)
        my_radius = max(1, int(bomb_bonus) + 1)

        enemies = []
        for i, p in enumerate(players):
            if i != self.agent_id and int(p[2]) == 1:
                enemies.append({
                    "pos": (int(p[0]), int(p[1])),
                    "radius": max(1, int(p[4]) + 1),
                })

        bombs = self._make_bombs(bombs_obs, players)
        bomb_positions = {b["pos"] for b in bombs}

        danger_at, block_until = self._simulate_danger(grid, bombs)
        first_danger = self._first_danger_time(my_pos, danger_at)

        # 1) ESCAPE: Immediate danger
        if self.escape_mode or first_danger <= 2:
            action = self._best_escape_action(grid, my_pos, block_until, danger_at, enemies)
            if action is not None:
                npos = self._next_pos(my_pos, action)
                if self._first_danger_time(npos, danger_at) > 5:
                    self.escape_mode = False
                return action
            return self._least_bad_action(grid, my_pos, block_until, danger_at)

        # 2) KILL PRIORITY
        if bombs_left > 0 and my_pos not in bomb_positions:
            kill_score = self._score_bomb_kill(grid, my_pos, enemies, my_radius)
            if kill_score >= 15 and self._can_escape_bomb(grid, my_pos, bombs, danger_at, my_radius):
                self.escape_mode = True
                return 5

        kill_action = self._move_to_kill_zone(grid, my_pos, enemies, block_until, danger_at)
        if kill_action is not None:
            return kill_action

        # 3) BOX FARMING
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes = self._count_boxes_in_blast(grid, my_pos, my_radius)
            if boxes >= 2 and self._can_escape_bomb(grid, my_pos, bombs, danger_at, my_radius):
                self.escape_mode = True
                return 5

        box_action = self._move_to_boxes(grid, my_pos, block_until, danger_at)
        if box_action is not None:
            return box_action

        # 4) ITEMS
        item_action = self._move_to_items(grid, my_pos, bombs_left, my_radius, block_until, danger_at)
        if item_action is not None:
            return item_action

        # 5) POSITIONING
        pos_action = self._move_to_safe_position(grid, my_pos, block_until, danger_at)
        if pos_action is not None:
            return pos_action

        return 0

    # =====================================================================
    # BOMBS AND DANGER SIMULATION
    # =====================================================================

    def _make_bombs(self, bombs_obs, players):
        """Parse bombs with owner radius info."""
        bombs = []
        for b in bombs_obs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            
            radius = 2
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            
            bombs.append({
                "pos": (bx, by),
                "timer": max(1, timer),
                "radius": radius,
            })
        return bombs

    def _simulate_danger(self, grid, bombs, horizon=15):
        """Simulate danger with chain reactions."""
        danger_at = [set() for _ in range(horizon + 1)]
        block_until = {}
        
        if not bombs:
            return danger_at, block_until

        blasts = [self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"]) for b in bombs]
        times = [b["timer"] for b in bombs]

        # Chain reaction: bomb gets triggered early if in another's blast
        changed = True
        iters = 0
        while changed and iters < len(bombs) + 3:
            changed = False
            iters += 1
            for i in range(len(bombs)):
                for j in range(len(bombs)):
                    if i != j and bombs[i]["pos"] in blasts[j]:
                        new_time = times[j] + 1
                        if times[i] > new_time:
                            times[i] = new_time
                            changed = True

        for i, b in enumerate(bombs):
            t = times[i]
            if 0 < t < len(danger_at):
                danger_at[t] |= blasts[i]
            block_until[b["pos"]] = t

        return danger_at, block_until

    def _blast_tiles(self, grid, bx, by, radius):
        """Calculate blast area."""
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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

    def _first_danger_time(self, pos, danger_at):
        """Find first danger time for position."""
        for t in range(1, len(danger_at)):
            if pos in danger_at[t]:
                return t
        return 999

    # =====================================================================
    # ESCAPE AND SAFETY
    # =====================================================================

    def _best_escape_action(self, grid, pos, block_until, danger_at, enemies, max_depth=8):
        """Find best escape action using BFS with safety scoring."""
        best_action = None
        best_score = -10**9
        
        q = deque([(pos, None, 0)])
        seen = {(pos, 0)}

        while q:
            cur_pos, first_action, dist = q.popleft()

            if dist > max_depth:
                continue

            # Score current position
            score = self._safety_score(cur_pos, dist, block_until, danger_at, grid)
            if score > best_score:
                best_score = score
                best_action = first_action

            if dist >= max_depth:
                continue

            # Explore neighbors
            for action in [1, 2, 3, 4]:
                npos = self._next_pos(cur_pos, action)
                if (npos, dist + 1) in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in danger_at[1]:
                    continue

                seen.add((npos, dist + 1))
                q.append((npos, action if first_action is None else first_action, dist + 1))

        return best_action if best_score > -10**8 else None

    def _safety_score(self, pos, turn, block_until, danger_at, grid):
        """Score position for safety."""
        score = 0.0
        
        # Check future danger
        first_danger = self._first_danger_time(pos, danger_at)
        if first_danger >= 999:
            score += 50
        else:
            score += max(0, first_danger - turn) * 2

        # Check open neighbors (mobility)
        opens = 0
        for a in [1, 2, 3, 4]:
            npos = self._next_pos(pos, a)
            if self._passable(grid, npos[0], npos[1]):
                opens += 1
        score += opens * 5

        return score

    def _least_bad_action(self, grid, pos, block_until, danger_at):
        """Fallback: any non-death action."""
        for a in [1, 2, 3, 4]:
            npos = self._next_pos(pos, a)
            if self._passable(grid, npos[0], npos[1]) and npos not in danger_at[1]:
                return a
        return 0

    def _can_escape_bomb(self, grid, pos, bombs, danger_at, my_radius):
        """Validate bomb placement allows escape."""
        my_blast = self._blast_tiles(grid, pos[0], pos[1], my_radius)
        
        # Check if at least one adjacent tile is safe
        for action in [0, 1, 2, 3, 4]:
            npos = self._next_pos(pos, action)
            if not self._passable(grid, npos[0], npos[1]):
                continue
            
            # Check safety for several future turns
            safe = True
            for t in range(1, min(8, len(danger_at))):
                if npos in danger_at[t] or npos in my_blast:
                    safe = False
                    break
            
            if safe:
                return True

        return False

    # =====================================================================
    # KILL PRIORITY
    # =====================================================================

    def _score_bomb_kill(self, grid, pos, enemies, radius):
        """Score bomb position for kill potential."""
        score = 0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)

        for enemy in enemies:
            if enemy["pos"] in blast:
                if self._line_clear(grid, pos, enemy["pos"]):
                    score += 25  # Direct hit
                else:
                    score += 10  # Indirect
        
        return score

    def _move_to_kill_zone(self, grid, pos, enemies, block_until, danger_at):
        """Move to position where we can threaten kills."""
        if not enemies:
            return None

        targets = {}
        for enemy in enemies:
            ex, ey = enemy["pos"]
            # Tiles adjacent to enemy
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tx, ty = ex + dx, ey + dy
                if self._passable(grid, tx, ty):
                    dist = abs(tx - pos[0]) + abs(ty - pos[1])
                    if dist <= 5:
                        targets[(tx, ty)] = 15 - dist

        if not targets:
            return None

        return self._move_bfs(grid, pos, targets, block_until, danger_at, max_dist=6)

    # =====================================================================
    # BOX FARMING
    # =====================================================================

    def _count_boxes_in_blast(self, grid, pos, radius):
        """Count destroyable boxes from position."""
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return sum(1 for x, y in blast if int(grid[x, y]) == 2)

    def _move_to_boxes(self, grid, pos, block_until, danger_at):
        """Move to high-value box positions."""
        targets = {}

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != 2:
                    continue

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) != pos:
                        dist = abs(nx - pos[0]) + abs(ny - pos[1])
                        if dist <= 8:
                            targets[(nx, ny)] = 10 - dist * 0.5

        if not targets:
            return None

        return self._move_bfs(grid, pos, targets, block_until, danger_at, max_dist=8)

    # =====================================================================
    # ITEM COLLECTION
    # =====================================================================

    def _move_to_items(self, grid, pos, bombs_left, bomb_radius, block_until, danger_at):
        """Move to high-value items."""
        targets = {}

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                dist = abs(x - pos[0]) + abs(y - pos[1])
                
                if cell == 3:  # Radius bonus
                    if bomb_radius <= 2:
                        targets[(x, y)] = 20 - dist
                    elif dist <= 6:
                        targets[(x, y)] = 5 - dist * 0.3
                        
                elif cell == 4:  # Capacity bonus
                    if bombs_left <= 2:
                        targets[(x, y)] = 20 - dist
                    elif dist <= 6:
                        targets[(x, y)] = 5 - dist * 0.3

        if not targets:
            return None

        return self._move_bfs(grid, pos, targets, block_until, danger_at, max_dist=10)

    # =====================================================================
    # POSITIONING AND FALLBACK
    # =====================================================================

    def _move_to_safe_position(self, grid, pos, block_until, danger_at):
        """Move to safe strategic position."""
        cx = (grid.shape[0] - 1) / 2.0
        cy = (grid.shape[1] - 1) / 2.0

        targets = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y):
                    continue
                
                dist_center = abs(x - cx) + abs(y - cy)
                dist_me = abs(x - pos[0]) + abs(y - pos[1])
                
                if dist_me <= 5:
                    targets[(x, y)] = 5 / (dist_center + 1)

        if not targets:
            return None

        return self._move_bfs(grid, pos, targets, block_until, danger_at, max_dist=5)

    # =====================================================================
    # PATHFINDING
    # =====================================================================

    def _move_bfs(self, grid, start, target_scores, block_until, danger_at, max_dist=12):
        """BFS to best scoring target, avoiding danger."""
        if not target_scores:
            return None

        q = deque([(start, None, 0)])
        seen = {start}
        best_action = None
        best_score = -10**9

        while q:
            pos, first_action, dist = q.popleft()

            if dist > max_dist:
                continue

            if pos in target_scores:
                score = target_scores[pos] - 0.05 * dist
                if score > best_score:
                    best_score = score
                    best_action = first_action

            if dist >= max_dist:
                continue

            for action in [1, 2, 3, 4]:
                npos = self._next_pos(pos, action)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                # Avoid immediate danger
                if any(npos in danger_at[t] for t in range(1, min(3, len(danger_at)))):
                    continue

                seen.add(npos)
                q.append((npos, action if first_action is None else first_action, dist + 1))

        return best_action if best_score > 0 else None

    # =====================================================================
    # UTILITIES
    # =====================================================================

    def _next_pos(self, pos, action):
        """Next position from action."""
        dx, dy = self.MOVES[action]
        return (pos[0] + dx, pos[1] + dy)

    def _in_bounds(self, grid, x, y):
        """Check bounds."""
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        """Check if tile is passable."""
        if not self._in_bounds(grid, x, y):
            return False
        cell = int(grid[x, y])
        return cell in [0, 3, 4]

    def _line_clear(self, grid, a, b):
        """Check line of sight."""
        ax, ay = a
        bx, by = b

        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if int(grid[ax, y]) in [1, 2]:
                    return False
            return True

        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if int(grid[x, ay]) in [1, 2]:
                    return False
            return True

        return False
