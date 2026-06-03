# Invinsible RL

Folder nay dung de train/test rieng `InvinsibleHybrid + mode-level Double DQN`.

## Chay agent

```powershell
.\.venv\Scripts\python.exe scripts\participant\estimate_rankings.py --agent_path agent\invinsible_rl --num_matches 100 --max_steps 500
```

`agent.py` se tu load `mode_model.pth` trong cung folder neu file do ton tai. Neu khong co model, agent tu fallback sang heuristic hybrid.

## Train mode-DQN

```powershell
.\.venv\Scripts\python.exe agent\invinsible_rl\train_mode.py --enemy_type weighted_mixed --num_episodes 500 --max_steps 500
```

Sau khi train, file model duoc luu tai:

```text
agent/invinsible_rl/mode_model.pth
```

Neu model lam agent yeu di, doi ten hoac xoa `mode_model.pth` de quay lai heuristic hybrid.
