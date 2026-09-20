# rl-from-scratch-C

A from-scratch reinforcement learning implementation in plain C — no external ML libraries. A small neural network (a policy) is trained with the **REINFORCE** algorithm to play Snake on a 6×6 grid, with a hand-rolled memory arena, matrix library, and automatic-differentiation (autograd) engine underneath it.

For a full plain-language walkthrough of how every part of this works, see [`snake_project_walkthrough.md`](./snake_project_walkthrough.md).

## What's in here

| File | Role |
|---|---|
| `base.h` | Shared types (`u32`, `f32`, ...) and small macros |
| `arena.c` / `arena.h` | Bump-allocator memory arena — all runtime memory lives here |
| `prng.c` / `prng.h` | PCG pseudo-random number generator |
| `mat.c` / `mat.h` | Matrix struct + operations (add, matmul, relu, softmax, ...) |
| `autograd.c` / `autograd.h` | Computation graph + forward/backward pass (backpropagation) |
| `model.c` | Builds the actual network: a 77 → 128 → 128 → 5 MLP policy |
| `env.c` | The Snake game itself, the REINFORCE training loop, and the live demo/play mode |

`env.c` is the entry point and `#include`s everything else in a single "unity build" — you only ever compile `env.c` directly.

## Requirements

- A C compiler (`gcc` or `clang`)
- The C math library (`-lm`)
- A terminal that supports ANSI color codes (for the live demo/play mode)

## Build

```bash
gcc -O2 -Wall -Wextra -o env env.c -lm
```

> **Note:** `-lm` must come *after* `env.c` on the command line. Most linkers resolve symbols in the order libraries are given, so `gcc -lm env.c` fails with `undefined reference to sqrtf/expf/logf`, while `gcc env.c -lm` works.

## Run

### Train from scratch

```bash
./env
```

Runs 3500 epochs of training (~a few minutes). Progress prints every 10 epochs:

```
Epoch 3499 | Avg Return:   77.04 | Foods: 133 | Samples:  2191
```

- **Avg Return** — average total reward per episode across the batch; should trend from negative to strongly positive.
- **Foods** — total food eaten across the 64-game batch that epoch; should climb steadily, especially past epoch ~1500.
- **Samples** — total game-steps recorded in the batch (a rough proxy for how long episodes are surviving).

When training finishes, it automatically saves `snake_weights.bin` and shows a short live demo.

### Watch a trained agent play

```bash
./env play
```

Loads `snake_weights.bin` (must already exist — run `./env` first if it doesn't) and plays 10 live episodes with colored terminal rendering:

- `@` — snake head
- `o` — snake body
- `*` — food

Expect the agent to eat food in the large majority of episodes, though not every single one — some starting layouts are just harder to solve than others.

## How training works, in one paragraph

The network looks at the snake's position, the food's position, and the direction it's currently facing, and outputs a probability for each of 5 moves (left, right, up, down, none). During training, moves are **sampled** from that distribution — not just the top pick — so the snake keeps exploring. After each batch of 64 games, moves that led to above-average outcomes get nudged to become more likely, and moves that led to below-average outcomes get nudged to become less likely, via backpropagation through the autograd engine in `autograd.c`. Repeat for 3500 batches and the probabilities converge toward "the moves that actually work." See the walkthrough doc for the full explanation, including the specific reward shaping used (`LIVING_PENALTY`, `DISTANCE_REWARD`, `FOOD_REWARD`, `STARVE_PENALTY_SCALE` in `env.c`).

## Known behavior / limitations

- The network sees only the snake's head, the food's position, and current facing direction — not the snake's body — so as the snake grows longer it has no direct signal to avoid its own tail.
- Training uses REINFORCE with a single batch-wide baseline (mean/std of returns), which is simple but higher-variance than more advanced policy-gradient methods.
- `./env play` uses greedy action selection most of the time, with a fallback to sampling if the agent goes too long without eating (see `test_time_action` in `env.c`) — this prevents the agent from getting stuck in a deterministic loop, but doesn't guarantee food is found in every episode.

## Project history

This code previously had a bug where the trained agent, when run in `./env play` mode, would frequently move in tight fixed loops and starve without eating — even though training itself worked correctly. The cause and fix are documented in detail in [`snake_project_walkthrough.md`](./snake_project_walkthrough.md#9-the-bug-why-the-snake-wasnt-eating).
