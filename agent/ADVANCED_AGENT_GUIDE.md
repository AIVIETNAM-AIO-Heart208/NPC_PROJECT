# 🚀 Hướng Dẫn Xây Dựng Agent Mạnh - Bomberland

**Mục tiêu**: Tạo một agent vượt trội hơn tất cả các baseline agent bằng cách khắc phục các điểm yếu cốt lõi.

---

## 📊 Phân Tích Các Baseline Agent

### 1. **RandomAgent** ❌
- Chỉ chọn hành động ngẫu nhiên [0-5]
- Không có chiến lược

### 2. **SimpleRuleAgent** ⚠️
- **Ưu điểm**: Tránh bom cơ bản, đặt bom đơn giản
- **Nhược điểm**: 
  - Chỉ sử dụng BFS cơ bản
  - Không dự đoán vị trí enemy
  - Không phân tích chuỗi nổ bom

### 3. **TacticalRuleAgent** ✅ (Mạnh nhất)
- **Ưu điểm**: 
  - Đặt bom có giá trị (targeting enemy, phá hộp)
  - Kiểm tra escape path
  - Adaptive priorities (items > boxes > pressure)
- **Nhược điểm**:
  - Chỉ tìm kiếm item trong vòng 10 bước
  - Không dự đoán chuyển động enemy
  - Không có lookahead sâu
  - Không tối ưu vị trí chiến lược

### 4. **BoxFarmerAgent** 🌾
- Tập trung vào items và boxes
- Quá passive, không aggressive

### 5. **GeniusRuleAgent** 🧠
- **Ưu điểm**: State-based escape tracking
- **Nhược điểm**: Vẫn reactive, không predictive

---

## 🔑 7 Điểm Yếu Chính Để Khắc Phục

### ❌ Vấn Đề 1: Không Dự Đoán Enemy
**Baseline**: Chỉ xem vị trí hiện tại của enemy
```python
# SAUII
enemies = [(int(p[0]), int(p[1])) for i, p in enumerate(players) if i != self.agent_id and p[2] == 1]
```

**Giải pháp**: 
- Mô phỏng 2-3 bước forward
- Tính toán reachable tiles cho mỗi enemy
- Predict safe/danger zones dựa trên prediction

### ❌ Vấn Đề 2: Lookahead Quá Nông
**Baseline**: Chỉ quyết định 1 bước, không mô phỏng kết quả

**Giải pháp**:
- Implement **alpha-beta pruning** hoặc **Minimax** đơn giản
- Mô phỏng 3-4 bước ahead cho bomb placement decisions
- Score các state dựa trên metrics

### ❌ Vấn Đề 3: Phân Tích Bomb Chain Yếu
**Baseline**: Chỉ check "có hit được enemy không" hoặc "có bao nhiêu box"

**Giải pháp**:
- Simulate bomb explosions (chain reactions)
- Tính toán tiles sẽ bị phá hủy
- Evaluate cleanup value (xóa sạch boxes)

### ❌ Vấn Đề 4: Assessment Tĩnh (Static Evaluation)
**Baseline**: Luôn ưu tiên items > boxes > pressure

**Giải pháp**:
- **Dynamic priorities** dựa trên game state:
  - Nếu đứng yên sẽ chết → escape ngay
  - Nếu enemy yếu → aggressive farming
  - Nếu items hiếm → collect tích cực
  - Nếu áp đảo → passive, survival

### ❌ Vấn Đề 5: Pathfinding Không Tối Ưu
**Baseline**: BFS thuần túy, không xem xét giá trị

**Giải pháp**:
- **Weighted pathfinding**:
  - Safety score (khoảng cách tới bom danger)
  - Item value (capacity > radius > boxes)
  - Enemy proximity penalty
- Sử dụng **A\* algorithm** với heuristic thông minh

### ❌ Vấn Đề 6: Không Quản Lý Resource
**Baseline**: Treat bombs và radius như nhau

**Giải pháp**:
- **Resource strategy**:
  - Bombs: Use strategically (kill > farm > survival)
  - Radius: Get ASAP → reach more boxes
  - Capacity: Secondary priority
- Evaluate trade-off giữa chúng

### ❌ Vấn Đề 7: Quá Reactive (Chỉ Phản Ứng)
**Baseline**: Chỉ escape khi danger, collect item, place bomb

**Giải pháp**:
- **Proactive play**:
  - Control key positions (center, item pickups)
  - Box farming efficiently
  - Create traps cho enemies
  - Maintain escape routes

---

## 🎯 Blueprint Agent Cải Tiến

### Cấu Trúc Core

```python
class AdvancedAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        # Hyperparameters
        self.SEARCH_DEPTH = 3  # Lookahead depth
        self.BOMB_TIMER = 7
        self.DEFAULT_BLAST_RADIUS = 1
        
        # Metrics/tuning
        self.DANGER_THRESHOLD = 6  # steps to explosion
        self.ITEM_VALUE = {"capacity": 100, "radius": 80}
        self.BOX_VALUE = 15
        self.KILL_ENEMY_VALUE = 500
        
    def act(self, obs):
        # 1. Parse observation
        grid, players, bombs = self._parse_obs(obs)
        my_pos = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
        
        # 2. Predict enemy movements
        enemy_predictions = self._predict_enemy_moves(grid, players, bombs)
        
        # 3. Calculate danger zones
        danger_now, danger_soon = self._calculate_danger(grid, bombs, players, enemy_predictions)
        
        # 4. Score all actions
        action_scores = {}
        for action in range(6):
            next_pos = self._get_next_pos(my_pos, action)
            if self._is_valid_move(grid, next_pos):
                score = self._evaluate_action(action, next_pos, grid, players, bombs, 
                                             danger_now, danger_soon, enemy_predictions)
                action_scores[action] = score
        
        # 5. Return best action
        return max(action_scores, key=action_scores.get)
```

### Phase 1: Safety First ⚠️
```
IF in immediate danger (danger_now):
   Find safest escape path
   Use predictive navigation (avoid predicted enemy positions)
   Return escape action
```

### Phase 2: Item Collection 🎁
```
IF not in danger AND items nearby:
   Calculate all item positions
   Weighted pathfinding to closest valuable item
   Return path action
```

### Phase 3: Strategic Bombing 💣
```
IF bombs available AND not in danger_soon:
   Simulate blast radius with chain reactions
   Evaluate: kill enemy value vs box farming value
   Check escape feasibility post-bomb
   IF value > threshold:
      Return PLACE_BOMB
```

### Phase 4: Position & Farm 🌾
```
IF no bombs, items far, enemies not immediate threat:
   Move to box farming positions (high box density)
   OR control map center
   Return strategic movement
```

---

## 🛠️ Implementasi Detail

### 1. **Predict Enemy Moves**

```python
def _predict_enemy_moves(self, grid, players, bombs, steps=3):
    """Predict enemy positions for next N steps"""
    predictions = {}
    for i, player in enumerate(players):
        if i == self.agent_id or player[2] != 1:  # dead or self
            continue
        
        enemy_pos = (int(player[0]), int(player[1]))
        # Predict: likely move away from danger / toward items/boxes
        reachable = self._bfs_reachable(grid, enemy_pos, bombs, max_steps=steps)
        predictions[i] = reachable
    
    return predictions

def _bfs_reachable(self, grid, pos, bombs, max_steps=3):
    """BFS to find all reachable tiles within max_steps"""
    from collections import deque
    queue = deque([(pos, 0)])
    visited = {pos}
    reachable = set()
    bomb_positions = {(b[0], b[1]) for b in bombs}
    
    while queue:
        (x, y), dist = queue.popleft()
        if dist > max_steps:
            continue
        reachable.add((x, y))
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and self._is_passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    
    return reachable
```

### 2. **Accurate Danger Calculation**

```python
def _calculate_danger(self, grid, bombs, players, enemy_predictions, default_radius=2):
    """Calculate danger zones with enemy movement prediction"""
    danger_now = set()
    danger_soon = set()
    
    for bomb in bombs:
        bx, by, timer, owner_id = int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3])
        blast_tiles = self._get_blast_tiles(grid, bx, by, default_radius)
        
        if timer <= 1:
            danger_now.update(blast_tiles)  # Explode next step
        if timer <= self.DANGER_THRESHOLD:  # e.g., 6 steps
            danger_soon.update(blast_tiles)
    
    # Add predicted enemy danger zones
    for enemy_id, reachable_tiles in enemy_predictions.items():
        for tile in reachable_tiles:
            danger_soon.add(tile)  # Enemy can reach this
    
    return danger_now, danger_soon

def _get_blast_tiles(self, grid, bx, by, radius):
    """Get all tiles affected by bomb at (bx, by)"""
    tiles = {(bx, by)}
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        for r in range(1, radius + 1):
            nx, ny = bx + dx*r, by + dy*r
            if not self._is_passable(grid, nx, ny):
                break
            tiles.add((nx, ny))
    return tiles
```

### 3. **Action Evaluation with Simulation**

```python
def _evaluate_action(self, action, next_pos, grid, players, bombs, 
                    danger_now, danger_soon, enemy_predictions):
    """Score an action by simulating it"""
    score = 0.0
    
    # Safety: +1000 for not in danger, -10000 for in danger_now
    if next_pos in danger_now:
        score -= 10000
    elif next_pos in danger_soon:
        score -= 500
    else:
        score += 100
    
    # Item collection
    item_value = self._evaluate_items_at(grid, next_pos, enemy_predictions)
    score += item_value
    
    # Bomb placement value
    if action == 5:  # PLACE_BOMB
        my_pos = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
        bomb_value = self._evaluate_bomb_placement(grid, my_pos, players, bombs, 
                                                   enemy_predictions, danger_soon)
        score += bomb_value
    
    # Lookahead: simulate 1-2 steps ahead
    lookahead_score = self._lookahead_evaluation(grid, next_pos, players, bombs, depth=2)
    score += lookahead_score * 0.5  # Weight future less than immediate
    
    return score

def _evaluate_items_at(self, grid, pos, enemy_predictions):
    """Value of items at position"""
    x, y = pos
    if grid[x, y] == 3:  # ITEM_RADIUS
        return self.ITEM_VALUE["radius"]
    elif grid[x, y] == 4:  # ITEM_CAPACITY
        return self.ITEM_VALUE["capacity"]
    return 0

def _evaluate_bomb_placement(self, grid, bomb_pos, players, bombs, enemy_predictions, danger_soon):
    """Evaluate value of placing bomb at bomb_pos"""
    score = 0.0
    blast_tiles = self._get_blast_tiles(grid, bomb_pos[0], bomb_pos[1], 2)  # assume radius=1 + 1 default
    
    # Box farming
    boxes = sum(1 for tile in blast_tiles if grid[tile[0], tile[1]] == 2)
    score += boxes * self.BOX_VALUE
    
    # Enemy kill potential
    for enemy_id, reachable in enemy_predictions.items():
        hit_prob = len(blast_tiles & reachable) / (len(reachable) + 1)
        score += hit_prob * self.KILL_ENEMY_VALUE
    
    # Check if can escape
    can_escape = self._can_escape_after_bomb(grid, bomb_pos, danger_soon)
    if not can_escape:
        score -= 5000  # Heavy penalty for self-kill
    
    return score
```

### 4. **Weighted A\* Pathfinding**

```python
def _best_path_to_target(self, grid, start, targets, bombs, danger_soon, weights=None):
    """Find best path to one of targets using A* with custom weights"""
    import heapq
    from math import sqrt
    
    if weights is None:
        weights = {"safety": 1.0, "distance": 1.0}
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in targets:
            return self._reconstruct_path(came_from, current)
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if not self._is_passable(grid, neighbor[0], neighbor[1]):
                continue
            if neighbor in [tuple(b[:2]) for b in bombs]:
                continue
            
            # Cost: distance + safety penalty
            move_cost = 1.0
            if neighbor in danger_soon:
                move_cost += 10.0  # Avoid danger
            
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Heuristic: distance to nearest target
                h_score = min(sqrt((neighbor[0]-t[0])**2 + (neighbor[1]-t[1])**2) for t in targets)
                f_score = tentative_g + h_score
                
                heapq.heappush(open_set, (f_score, neighbor))
    
    return None  # No path found
```

---

## 🎮 Testing & Tuning

### 1. **Local Testing**
```bash
# Test against baselines
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ \
  TacticalRuleAgent \
  GeniusRuleAgent \
  RandomAgent \
  --visualize true

# Estimate rankings
python -m scripts.participant.estimate_rankings --agent_path agent/advanced_agent/ --num_matches 50
```

### 2. **Tuning Hyperparameters**

Start with:
- `SEARCH_DEPTH = 2` (increase only if still within 100ms)
- `DANGER_THRESHOLD = 6`
- `KILL_ENEMY_VALUE = 500`
- `BOX_VALUE = 15`

Measure win rates against each baseline, adjust:
- If vs Random too slow: reduce depth
- If vs Tactical loses: increase bomb placement value
- If vs Genius loses: improve escape logic

---

## 📝 Development Checklist

- [ ] Implement basic agent structure
- [ ] Add enemy position prediction (BFS reachable)
- [ ] Implement accurate danger zone calculation
- [ ] Add bomb chain reaction simulation
- [ ] Implement action evaluation with scoring
- [ ] Add A\* pathfinding with weights
- [ ] Integrate lookahead evaluation
- [ ] Test locally vs each baseline
- [ ] Optimize for 100ms time limit
- [ ] Tune hyperparameters based on win rates
- [ ] Package and submit

---

## 💡 Optimizations Nâng Cao (Advanced)

Once baseline works well:

1. **Memoization**: Cache blast zones, reachability
2. **Multi-threading**: Evaluate multiple bomb positions in parallel
3. **Game tree search**: Implement proper minimax for critical decisions
4. **Learning**: Track what worked/failed, adjust weights dynamically
5. **Cooperative play**: Model ally positions (even simple models help)
6. **Late-game strategy**: Different behavior when 1-2 players remain

---

## 🚀 Submission

1. Create folder `agent/advanced_agent/`
2. Save your agent as `agent.py` with class `Agent`
3. If using weights: save in same folder, load with `Path(__file__).parent`
4. Zip and submit via form

---

**Good luck! 🎯**
