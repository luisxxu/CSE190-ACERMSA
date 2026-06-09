# DPAMSA Analysis

An extension of **DPAMSA** (Deep-reinforcement-learning Profile-based Multiple Sequence Alignment) from [Liu et al. 2023](https://academic.oup.com/bioinformatics/article/39/11/btad636/7323576) from the [DPAMSA repo](https://github.com/ZhangLab312/DPAMSA) that adds new RL algorithms, training/runtime optimizations, and a benchmarking pipeline comparing the learned aligners against traditional MSA tools (MAFFT, ClustalW, Clustal Omega, MSAProbs).

## Summary of changes

- **More algorithms.** Alongside the original DQN, added A2C, PPO, and ACER agents that share the same Transformer encoder and environment (`--algorithm` flag).
- **Faster, more stable training.** Double DQN, Huber loss + gradient clipping, prioritized experience replay, multi-head attention, mixed-precision (`torch.cuda.amp`), `torch.compile`, and tuned hyperparameters.
- **Parallel environments.** `parallel_env.py` runs N environments per step for higher data throughput (automatic when `n_envs > 1` in `config.py`).
- **Better embeddings.** Unique token IDs for IUPAC ambiguity codes (N/R/W/K/Y) and `√d_model` embedding scaling.
- **Profile-based scoring.** PSSM-based `calc_profile_score()` computes an SP-equivalent score in O(L·k) instead of O(L·k²) (`--scoring`).
- **Benchmarking.** Per-test CSV + figure generation comparing each algorithm against traditional MSA tools.

Full per-file rationale lives in the project report.

## Setup

```bash
bash setup_env.sh        # creates the conda env from environment.yml
conda activate dpamsa    # (or the env name printed by the script)
```

MAFFT is fetched automatically by `_ensure_mafft.sh` when running the benchmark scripts. See the report for the other traditional MSA tools and the versions used.

## Running

### Align a single dataset
`main.py` takes a FASTA file and an algorithm. Models are saved/loaded under `weights/`.

```bash
python main.py sequences.fasta                                  # DQN (default)
python main.py sequences.fasta --algorithm acer                 # ACER
python main.py sequences.fasta --algorithm ppo --scoring both   # PPO, print SP + profile scores
python main.py sequences.fasta --algorithm acer --episodes 5000 --save my_model
```

Key flags: `--algorithm {dqn,a2c,ppo,acer}`, `--scoring {sp,profile,both}`, `--episodes`, `--save/--load NAME`, `--results-csv`, `--figures-dir`. Edit `config.py` for hyperparameters and `n_envs` (parallel training).

### Run the benchmarks
The `run_acer_*.sh` scripts train ACER across a full dataset and emit a results CSV + figures. They take an optional start/end test range:

```bash
bash run_acer_3x30bp.sh          # all tests
bash run_acer_6x30bp.sh 10 19    # resume from test10 to test19
bash run_acer_6x60bp.sh
```

### Hyperparameter grid search

```bash
python grid_search.py --dataset 3x30 --n_tests 10 --episodes 60
```

## Results

- `luis_results/` — benchmark CSVs (`acer_3x30bp`, `acer_6x30bp`, `acer_6x60bp`, `grid_search`).
- `figures/luis_figures/` — grouped-bar, delta, win-rate, and summary figures per dataset.
