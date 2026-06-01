# 🔧 Tuning Guide & Debugging - Advanced Agent

## Quick Start to Testing

### 1. First Test (Sanity Check)
```bash
cd c:\Python\Bomberland\Bomberland-GDGoC-AI-Challenge

# Play one visual match against Random
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ None None None \
  --visualize true
```

**Expected**: Your agent should move and avoid obvious bombs.

### 2. Baseline Comparison
```bash
# vs Random (should win 95%+)
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ RandomAgent None None \
  --visualize false

# vs Simple (should win 70%+)
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ SimpleRuleAgent None None \
  --visualize false

# vs Tactical (should win 55%+)
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ TacticalRuleAgent TacticalRuleAgent None \
  --visualize false
```

### 3. Estimate Rating
```bash
python -m scripts.participant.estimate_rankings \
  --agent_path agent/advanced_agent/ \
  --num_matches 50
```

---

## 🎯 Performance Targets

| Baseline | Expected Win Rate | If Not Hitting → Adjust |
|----------|------------------|--------------------------|
| Random | 95%+ | Agent has bug (should always win) |
| Simple | 70%+ | Improve bomb placement logic |
| Tactical | 55%+ | Improve lookahead/prediction |
| Genius | 50-60% | Fine-tune hyperparameters |

---

## 🐛 Common Issues & Fixes

### Issue 1: Agent Crashes
**Symptom**: "Agent error:" in logs

**Debug**:
```python
# Add this in act() method:
try:
    # ... existing code ...
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print(traceback.format_exc())
    return self.STOP
```

**Common causes**:
- Array index out of bounds → Check player validation
- None returned from function → Add fallback returns
- Division by zero → Add +0.1 to denominators

---

### Issue 2: Too Slow (Exceeds 100ms)
**Symptom**: Match hangs or agent gets "timeout"

**Solutions**:
1. Reduce `SEARCH_DEPTH` to 1
2. Limit enemy prediction steps to 2
3. Use simpler lookahead
4. Precompute blast radiuses

```python
# Optimize blast tile calculation
def _get_blast_tiles(self, grid, bx, by, radius):
    # CACHE if called multiple times per step
    if not hasattr(self, '_blast_cache'):
        self._blast_cache = {}
    
    key = (bx, by, radius)
    if key in self._blast_cache:
        return self._blast_cache[key]
    
    tiles = {(bx, by)}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for r in range(1, radius + 1):
            nx, ny = bx + dx * r, by + dy * r
            if not self._is_passable(grid, nx, ny):
                break
            tiles.add((nx, ny))
    
    self._blast_cache[key] = tiles
    return tiles
```

---

### Issue 3: Wins vs Random but Loses to Tactical
**Symptom**: Good against weak agents, bad against smart ones

**Analysis**: Your lookahead/bomb evaluation is weak

**Fixes**:
```python
# Increase bomb value
self.KILL_ENEMY_VALUE = 600  # was 400
self.BOX_VALUE = 20  # was 15

# Improve bomb escape check
def _has_safe_escape(self, grid, bomb_pos, bomb_positions, danger_soon, steps=5):  # was 3
    # ... existing code ...
```

---

### Issue 4: Moves Into Danger
**Symptom**: Agent walks into bombs

**Debug**: Check danger calculation
```python
# Print debug info
def _calculate_danger(self, grid, bombs, players, enemy_predictions, default_radius=2):
    danger_now, danger_soon = ...
    
    # Verify calculations
    assert danger_now.isdisjoint(danger_soon) == False  # overlap OK
    assert len(danger_now) <= len(danger_soon)  # soon >= now
    
    return danger_now, danger_soon
```

---

### Issue 5: Never Places Bombs
**Symptom**: Only moves, never attacks

**Debug**:
```python
# Check bomb availability
print(f"Bombs left: {bombs_left}")
print(f"Position occupied: {self._is_occupied_by_bomb(bombs, my_pos)}")
print(f"Bomb value: {bomb_value}")
print(f"Can escape: {self._has_safe_escape(...)}")
```

**Likely cause**: `_has_safe_escape()` always returns False → reduce `steps` parameter or relax danger threshold

---

## 📊 Hyperparameter Tuning Guide

### Safety Parameters
```python
# Increase if agent dies too much
self.DANGER_THRESHOLD = 7  # Steps before danger (was 6)
self.SEARCH_DEPTH = 3      # Lookahead depth (was 2)

# Decrease if too passive
self.IMMEDIATE_DANGER_PENALTY = -5000  # was -10000
```

### Offensive Parameters
```python
# Increase if losing to Tactical
self.KILL_ENEMY_VALUE = 500  # was 400
self.BOX_VALUE = 20           # was 15

# Decrease if self-destructing
self.KILL_ENEMY_VALUE = 300
```

### Positional Parameters
```python
# Increase if want more center control
self.POSITION_CONTROL_VALUE = 10  # was 5

# Decrease if want more farming
self.POSITION_CONTROL_VALUE = 2
```

### Item Parameters
```python
# Prioritize radius early
self.ITEM_RADIUS_VALUE = 150   # was 80
self.ITEM_CAPACITY_VALUE = 100

# Or opposite for endgame
self.ITEM_CAPACITY_VALUE = 150
```

---

## 🎮 Win Rate Improvement Plan

### Step 1: Beat Random (Target: 95%+)
✅ Should work out of the box
- If not: debug crashes first

### Step 2: Beat Simple (Target: 70%+)
- Improve bomb placement: increase `BOX_VALUE`
- Better escaping: check `_has_safe_escape()` logic

**Tuning**:
```python
# In agent init
self.BOX_VALUE = 20  # Increase interest in farming
self.KILL_ENEMY_VALUE = 300  # Less aggressive
```

### Step 3: Beat Tactical (Target: 55%+)
- Improve lookahead
- Better enemy prediction
- Smarter bomb timing

**Tuning**:
```python
# Increase lookahead sophistication
self.SEARCH_DEPTH = 3
self.DANGER_THRESHOLD = 7

# Better bomb placement
self.KILL_ENEMY_VALUE = 400
self.BOX_VALUE = 25
```

### Step 4: Beat Genius (Target: 50%+)
- Fine-tune escape logic
- Optimize resource management
- Reduce timeout penalties

---

## 📈 Profiling & Optimization

### Measure Execution Time
```python
import time

def act(self, obs):
    start = time.time()
    
    # ... your code ...
    
    elapsed = (time.time() - start) * 1000  # milliseconds
    if elapsed > 50:  # Start warning at 50ms
        print(f"WARNING: Act took {elapsed:.1f}ms")
    
    return best_action
```

### Bottleneck Identification
```python
# Likely slow parts:
# 1. Enemy prediction (_bfs_reachable) - O(reachable cells)
# 2. Action evaluation - O(6 actions)
# 3. Blast tile calculation - O(radius * 4 directions)

# Optimize by:
# - Caching blast zones
# - Reducing prediction depth to 2
# - Early action elimination
```

---

## 🧪 Advanced Debugging

### Print State During Match
```python
def act(self, obs):
    if obs["bombs"]:
        print(f"\n=== STEP {random.randint(0, 1000)} ===")
        print(f"My pos: {(int(obs['players'][self.agent_id][0]), int(obs['players'][self.agent_id][1]))}")
        print(f"Bombs: {len(obs['bombs'])}")
        print(f"Action scores: {action_scores}")
```

### Replay Specific Match
```bash
# If you saved a match log
python -m scripts.participant.replay_viewer path/to/match.json

# Or generate detailed log
python -m scripts.participant.run_local_match \
  --agent_paths agent/advanced_agent/ TacticalRuleAgent None None \
  --visualize true \
  --save_replay true  # If supported
```

---

## 🚀 Advanced Improvements

### Once baseline works well (50%+ vs Tactical):

#### 1. Multi-Agent Cooperation
```python
def _get_ally_positions(self, players):
    """Identify friendly players (simple heuristic)"""
    # In a real competition, you'd track this differently
    # For now, just know positions to not collide
    return [
        (int(p[0]), int(p[1])) 
        for i, p in enumerate(players)
        if i != self.agent_id and p[2] == 1
    ]
```

#### 2. Game Tree Minimax
```python
def _minimax(self, state, depth, is_maximizing, alpha, beta):
    """Simple alpha-beta pruned minimax"""
    if depth == 0:
        return self._evaluate_state(state), None
    
    if is_maximizing:
        max_eval = float('-inf')
        best_move = None
        for action in range(6):
            new_state = self._simulate_move(state, action)
            eval_score, _ = self._minimax(new_state, depth - 1, False, alpha, beta)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Prune
        return max_eval, best_move
    else:
        # Minimizing (enemies' best counter-moves)
        ...
```

#### 3. Machine Learning Layer
```python
# Track outcomes and learn
class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.action_history = []
        self.reward_history = []
        
    def learn_from_match(self, outcomes):
        """Update weights based on match results"""
        # Could implement simple value updates
        pass
```

---

## 📋 Testing Checklist

- [ ] Agent runs without crashes
- [ ] Wins vs Random (95%+)
- [ ] Wins vs Simple (70%+)
- [ ] Wins vs Tactical (55%+)
- [ ] Time per action < 100ms
- [ ] No infinite loops
- [ ] Handles edge cases (alone, cornered, no items)
- [ ] Bomb placement has good value
- [ ] Escaping logic is solid
- [ ] Item collection works
- [ ] Ready for submission ✅

---

## 🎯 Final Tips

1. **Test incrementally** - Add one feature, test, then next
2. **Keep it simple** - Complex ≠ better (check time limits)
3. **Profile early** - Know what's slow before optimizing
4. **Tune systematically** - Change one param at a time
5. **Document changes** - Track what improves/hurts win rate

**Good luck! Your agent should beat all baselines with these guidelines! 🚀**
