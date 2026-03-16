<p align="center">
  <img src="mosquito.png" alt="FlyQuest Logo" width="200"/>
</p>

# FlyQuest 🪰

A Deep Q-Network agent trained to hunt a moving target in a 2D grid world. Built in PyTorch with experience replay, a target network, and a Prometheus + Grafana observability stack.

## Evaluation Results

| Episodes | Win Rate | Spider Deaths | Timeouts |
| :---: | :---: | :---: | :---: |
| **10,000** | **84.09%** | **15.89%** | **0.02%** |

---

## Getting Started

**Train:**
```bash
pip install -r requirements.txt
python train.py
```

**Evaluate:**
```bash
python eval.py
```

**Full stack with observability:**
```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (password: `flyq`) |

---

## The Journey

This project wasn't built in one shot. Each phase below exposed a real failure, diagnosed its root cause, and applied a principled fix.

### Phase 1 — Tabular Q-Learning

Built a Q-Table as a Python dictionary mapping `(state) → [action scores]`. Implemented the Bellman equation to propagate rewards backward over time. Used epsilon-greedy exploration — the agent starts fully random and gradually shifts toward its learned policy.

### Phase 2 — The Curse of Dimensionality

With a stationary Human the state space had 100 states. Making the Human move randomly exploded it to **10,000 states**. Adding moving Spiders would produce **10 billion** — more RAM than physically exists. The tabular approach hit its ceiling.

### Phase 3 — Deep Q-Network

Replaced the dictionary with a neural network that generalizes rather than memorizes.

```
Input:   10 neurons  (fly, human, 3 spider positions)
Hidden:  64 → 64     (ReLU)
Output:  4 neurons   (Q-values for Up, Left, Down, Right)
```

### Phase 4 — Experience Replay

Training one step at a time causes catastrophic forgetting — the network overwrites old lessons with new ones. Storing 10,000 past experiences in a replay buffer and sampling random batches of 64 breaks the temporal correlation and stabilizes learning.

### Phase 5 — The Oscillation Bug

After 10,000 episodes the Fly got stuck bouncing against walls instead of chasing the Human. Three root causes:

- **No step limit** — early random episodes ran for thousands of steps, flooding the buffer with wall-bouncing data. The network learned to bounce because that was 99% of what it saw.
- **No target network** — using the same network to both predict Q-values and compute Bellman targets means every weight update shifts the "correct answer." The goalposts never stop moving.
- **No timeout penalty** — failing to catch the Human had no cost, so the agent had no signal that stalling was bad.

**Fixes:** `max_steps = 200`, a `-10` timeout penalty, and a frozen target network refreshed every 100 training steps.

### Phase 6 — The Blind Spider Problem

The Fly continued dying erratically. The state `(fly_x, fly_y, human_x, human_y)` gave the network zero information about spider positions. The agent was navigating a minefield blindfolded.

**Fix:** Expanded state to 10 inputs by appending all 3 spider coordinates. Used curriculum learning — trained without spiders first until the agent reliably hunted the Human, then reintroduced spiders with the expanded state.

### Phase 7 — The Reward Imbalance Bug

With spiders visible, the Fly started spamming one direction into a wall indefinitely. The spider penalty (`-100`) was 10× the human reward (`+10`). The network found a rational but useless local minimum: press into a wall forever — zero spider risk at the cost of `-1` per step.

**Fixes:** Balanced the spider penalty to `-10`. Added a `-2` wall penalty. Introduced reward shaping — `-0.5` for moving closer to the Human, `-1.5` for moving away — giving the network dense directional feedback on every step instead of only at episode end.

### Phase 8 — Stable Policy

After applying all fixes, the agent converged to a stable policy with 84% win rate over 10,000 evaluation episodes with zero exploration (epsilon = 0.0).
