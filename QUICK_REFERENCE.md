# 🎯 Quick Reference - Advanced Agent Implementation

## 📋 Implementation Checklist

### Phase 1: Setup ✅
- [x] Created `agent/advanced_agent/` folder
- [x] Implemented core `Agent` class
- [x] Added helper functions

**Next**: Test it!

### Phase 2: Testing 🔄
- [ ] Run against Random (expect 95%+)
- [ ] Run against Simple (expect 70%+)
- [ ] Run against Tactical (expect 55%+)
- [ ] Check execution time (<100ms)

### Phase 3: Debugging 🔧
- [ ] If crashes: add exception handling
- [ ] If too slow: reduce SEARCH_DEPTH
- [ ] If loses to Random: find bug
- [ ] If loses to Simple: tune BOX_VALUE

### Phase 4: Tuning 📈
- [ ] Adjust hyperparameters per TUNING_GUIDE.md
- [ ] Run 20+ matches per configuration
- [ ] Track win rates
- [ ] Find optimal parameters

### Phase 5: Optimization 🚀
- [ ] Cache expensive calculations
- [ ] Profile bottlenecks
- [ ] Implement advanced features
- [ ] Final testing

### Phase 6: Submission ✈️
- [ ] Package: `agent.py` + any weights
- [ ] Zip folder flat (no nested folders)
- [ ] Submit via official form

---

## 🎮 Test Commands

### Quick Test (Visual)
```bash
cd c:\Python\Bomberland\Bomberland-GDGoC-AI-Challenge
python -m scripts.participant.run_local_match --agent_paths agent/advanced_agent/ RandomAgent None None --visualize true
```
**Expected**: Your agent moves around, avoids bombs

### Quick Test (Headless)
```bash
python -m scripts.participant.run_local_match --agent_paths agent/advanced_agent/ SimpleRuleAgent None None --visualize false
```
**Expected**: Should win most matches

### Rating Estimate
```bash
python -m scripts.participant.estimate_rankings --agent_path agent/advanced_agent/ --num_matches 50
```
**Expected**: Shows win rate and TrueSkill rating

---

## 📊 Architecture Overview

```
OBSERVATION (game state)
    │
    ├─→ [1] PREDICTION
    │       └─ Enemy reachable positions (3 steps)
    │
    ├─→ [2] DANGER CALCULATION
    │       ├─ Bomb danger zones
    │       └─ Enemy predicted zones
    │
    ├─→ [3] ACTION SCORING (for each action 0-5)
    │       ├─ Safety score (danger_now/soon)
    │       ├─ Item value (radius/capacity)
    │       ├─ Bomb value (boxes + enemies)
    │       ├─ Position value (center control)
    │       └─ Lookahead value (1 step ahead)
    │
    └─→ [4] BEST ACTION
            └─ Return action with highest score
```

---

## 🔢 Key Metrics to Track

When testing, record:

```
Agent: advanced_agent
Opponent: [RandomAgent, SimpleRuleAgent, TacticalRuleAgent, GeniusRuleAgent]

Matches: 20
Wins: 19
Win Rate: 95%

Avg Death Turn: 350 (higher = survives longer)
Bombs Placed: 8.5 (per match)
Items Collected: 2.3 (per match)
Kills: 0.7 (per match)
```

**Analysis**:
- If win rate too low: Tune hyperparameters
- If bombs placed = 0: Fix bomb placement logic
- If deaths early: Fix danger calculation
- If kills = 0: Increase KILL_ENEMY_VALUE

---

## 🛠️ Common Tuning Scenarios

### Scenario 1: Loses to Random
**Problem**: Your agent is broken
**Solution**: 
- Check for exceptions in act()
- Verify action returns are valid (0-5)
- Test with visualize=true to watch behavior

### Scenario 2: Wins Random, Loses Simple
**Problem**: Basic strategy weak
**Solution**:
```python
# Increase interest in farming
self.BOX_VALUE = 25  # was 15
self.ITEM_RADIUS_VALUE = 100  # was 80
```

### Scenario 3: Wins Simple, Loses Tactical
**Problem**: Lookahead/planning weak
**Solution**:
```python
# Deeper search
self.SEARCH_DEPTH = 3  # was 2
self.DANGER_THRESHOLD = 7  # was 6

# Better bomb placement
self.KILL_ENEMY_VALUE = 450  # was 400
```

### Scenario 4: Wins Tactical, Loses Genius
**Problem**: Escape logic weak
**Solution**:
```python
# Better escape evaluation
def _has_safe_escape(self, ..., steps=5):  # was 3
    # Now checks further ahead

# More conservative
self.NEAR_DANGER_PENALTY = -1000  # was -500
```

---

## ⚡ Performance Tuning

### If Too Slow (>80ms per action)

**Priority 1**: Reduce predictions
```python
def _predict_enemy_moves(self, grid, players, bombs, steps=2):  # was 3
```

**Priority 2**: Reduce lookahead
```python
self.SEARCH_DEPTH = 1  # was 2
```

**Priority 3**: Cache blast zones
```python
# In __init__
self._blast_cache = {}

# In _get_blast_tiles (add caching)
if key in self._blast_cache:
    return self._blast_cache[key]
```

**Priority 4**: Early action pruning
```python
# Skip obviously bad actions early
for action in range(6):
    if action == 5 and bombs_left == 0:
        continue  # Can't bomb
    if next_pos in danger_now:
        continue  # Can't go there
```

### If Fast Enough (<50ms)

**Now optimize quality**:
- Increase SEARCH_DEPTH to 3
- Better lookahead evaluation
- Implement minimax
- Add memoization

---

## 🎯 Competitive Win Rates

### Minimum for Submission
- Random: 90%+ ✅
- Simple: 60%+ ⚠️
- Tactical: 45%+ ⚠️

### Target for Competitive
- Random: 98%+ ✅
- Simple: 75%+ ✅
- Tactical: 55%+ 🎯
- Genius: 50%+ 🎯

### Elite Performance
- Random: 99%+ ✅
- Simple: 85%+ ✅
- Tactical: 65%+ 🎯
- Genius: 60%+ 🎯

---

## 📁 File Organization

```
Bomberland-GDGoC-AI-Challenge/
├── agent/
│   ├── ADVANCED_AGENT_GUIDE.md     📖 Strategy & theory
│   ├── TUNING_GUIDE.md             🔧 Debugging & optimization
│   ├── advanced_agent/
│   │   ├── agent.py                🤖 Main implementation
│   │   ├── __init__.py
│   │   └── README.md               📚 Quick reference
│   │
│   ├── tactical_rule_agent.py      (baseline to beat)
│   ├── genius_rule_agent.py        (baseline to beat)
│   └── ...
│
└── scripts/participant/
    ├── run_local_match.py          🎮 Test matches
    ├── estimate_rankings.py        📊 Rating estimation
    └── replay_viewer.py            🎬 Watch replays
```

---

## 🚀 Submission Workflow

### 1. Prepare Files
```bash
cd agent/advanced_agent/
ls
# Should show: agent.py, __init__.py, README.md
```

### 2. Package for Submission
```bash
# Zip the folder contents (not the folder itself)
# Windows: Right-click → Send to → Compressed folder
# Or: tar czf advanced_agent.zip agent.py __init__.py README.md

# Result: advanced_agent.zip containing:
# - agent.py (with class Agent)
# - __init__.py (if using imports)
# - any weight files (if using ML)
```

### 3. Submit
- Go to official submission form
- Upload .zip file
- Enter Team ID
- Enter Token
- Submit!

### 4. Monitor
- Agent plays 12 matches immediately
- Check leaderboard for rating
- Get feedback in 2-3 hours

---

## 📞 Troubleshooting

| Issue | Command to Debug | Expected Output |
|-------|------------------|-----------------|
| Agent crashes | `--visualize true` | See error on screen |
| Too slow | Add timing prints | <100ms per step |
| Loses to Random | Check `act()` returns | Valid action 0-5 |
| Never bombs | Print `bombs_left` | Should > 0 sometimes |
| Moves into danger | Print danger zones | Should avoid them |

---

## 📈 Performance Progression

Typical improvement over time:

```
Week 1: Get basic agent working (50% vs Random)
        ↓
Week 2: Fix bugs, reach 90% vs Random
        ↓
Week 3: Tune parameters, 70% vs Simple
        ↓
Week 4: Optimize lookahead, 55% vs Tactical
        ↓
Week 5: Fine-tune, 50%+ vs Genius
        ↓
Submit: Ready for competition! 🎉
```

---

## ✅ Final Checklist Before Submission

- [ ] Agent runs without crashes
- [ ] Beats Random reliably (95%+)
- [ ] Beats Simple (70%+)
- [ ] Competitive vs Tactical (55%+)
- [ ] Execution time <100ms
- [ ] Code is clean and commented
- [ ] Files organized properly
- [ ] Tested locally 50+ times
- [ ] Ready to submit!

---

## 🎓 Learning Resources

**Inside This Repo**:
- [ADVANCED_AGENT_GUIDE.md](ADVANCED_AGENT_GUIDE.md) - Deep dive
- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Practical debugging
- `agent/` folder - All baseline agents to learn from

**External Resources**:
- A* Search: https://en.wikipedia.org/wiki/A*_search_algorithm
- Minimax: https://en.wikipedia.org/wiki/Minimax
- BFS: https://en.wikipedia.org/wiki/Breadth-first_search

**Competition Resources**:
- See `README.md` in main folder
- Join Discord (if available)
- Check leaderboard regularly

---

**You're all set! Test, tune, and submit! 🚀**

Questions? Check TUNING_GUIDE.md or ADVANCED_AGENT_GUIDE.md for detailed answers.
