## 1  Project overview

The **Stock Prediction Toolkit** is a research sandbox for portfolio-selection methods.  
It lets you:

* **Classical optimisation** – Naive Markowitz, GMVP, CA-GMVP  
* **Reinforcement learning** – MarginTrader (A2C) with multi-asset action space  
* **Head-to-head comparison** – identical price data, shared cost model, same look-back window  
* **One-command visualisation** – overlay equity curves & export publication-ready figures

The typical workflow is:

1. Load the provided `Database.csv` (2 884 dates × 1 224 tickers).  
2. Train each model with consistent hyper-parameters.  
3. Aggregate Sharpe, max drawdown, turnover, etc. across models.  
4. Plot equity curves and build a slide deck for presentation.


## 1  Project Layout

```
.
├─ data/                 ← Database.csv & any prepared datasets
├─ models/
│   ├─ MarginTrader/     ← RL (A2C) agent
│   ├─ NaiveMarkowitz/
│   ├─ GMVP/
│   └─ CA-GMVP/
├─ utils/
│   ├─ compare_all.py    ← aggregates every model’s metrics
│   └─ plot_equity.py    ← equity-curve visualiser
├─ requirements.txt
└─ README.md             ← (this file)
```

## 2  Set up
<details> <summary><b>Option A – GPU / Conda <i>(recommended)</i></b></summary>
# 1  Create env (Python 3.12 – rename if you like)  
conda create -n stock-gpu python=3.12 -y  
conda activate stock-gpu  

# 2  Install the correct PyTorch wheel for YOUR CUDA version
###     (CUDA 12.1 wheel runs fine on driver ≥570; swap cu118 / cu120 … if needed)
pip install torch==2.2.2+cu121 \
             --extra-index-url https://download.pytorch.org/whl/cu121

# 3  Rest of the stack
pip install -r requirements.txt
</details> <details> <summary><b>Option B – CPU-only <i>(runs everywhere, slower)</i></b></summary>
python -m venv .venv && source .venv/bin/activate  
pip install --upgrade pip  
pip install torch==2.2.2+cpu  
pip install -r requirements.txt  
</details>

## 3  Quick Start Workflow
### 0  Move to repo root
cd "Stock Prediction Toolkit"

### 1  Train every model (each script writes to results/)
python models/MarginTrader/train.py          # RL (GPU if available)  
python models/NaiveMarkowitz/train.py  
python models/GMVP/train.py  
python models/CA-GMVP/train.py  

### 2  Compare them side-by-side (Sharpe, drawdown, etc.)
python utils/compare_all.py                  # → results/summary.csv

### 3  Visualise equity curves
python utils/plot_equity.py                  # shows window or saves PNG

## 4  Algorithms implemented

| Folder / script                               | Type | One-liner intuition | Main hyper-parameters you can tune |
|-----------------------------------------------|------|---------------------|------------------------------------|
| **MarginTrader** (`models/MarginTrader/`)     | **RL – A2C** | Learns continuous long/short **weights ∈ [-1, 1]²ᴺ**. Account is rescaled to ≤ 1.5 × equity each step and pays 20 bp round-trip cost. Observations = cash/equity/credit + current price vector + holdings. We log **RawReturn** (can be ≤ –100 %) and a clipped series used only for NAV. | `total_steps`, `n_steps`, `γ`, `ent_coef`, `penalty`, `lr`, `max_leverage` |
| **Naïve Markowitz** (`models/NaiveMarkowitz/`) | **Mean-variance optimiser** | maximise μᵀw − η·wᵀΣw with ∑w = 1 (shorts optional), rebalanced every 5 days on a 252-day window. | `eta` (risk-aversion), `shorts` on/off |
| **GMVP** (`models/GMVP/`)                     | **Global Minimum-Variance** | minimise wᵀΣw subject to ∑w = 1; ignores expected returns. | *(no extra params)* |
| **CA-GMVP Clustering** (`models/GMVP_Clustering/`) | **Cost-Aware GMVP** | Adds ℓ₁/ℓ₂ turnover penalties and soft short limit, clusters stocks (≤ `max_cluster`/cluster) to stabilise Σ; solved with CVXPY. | `n_clusters`, `max_cluster`, turnover weight |

All four models see **exactly the same price matrix, 20 bp transaction cost, 252-day look-back, and 5-day re-balance cadence** so performance differences come from strategy logic alone.

---

## 5  Common experimental constants

| Constant | Value | Used by | Why it matters |
|----------|-------|---------|----------------|
| **LOOKBACK** | **252 trading days** | all models | One full year of history for μ and Σ estimates. |
| **EVAL_WINDOW** | **5 days** | all models | Weekly re-balance to keep turnover realistic. |
| **TRANSACTION_COST** | **0.002** (20 bp round trip) | all models | Uniform cost so RL and optimisers pay the same friction. |
| **INITIAL_CAPITAL** | **1 000 000 USD** | *MarginTrader + equity replay only* | Sets starting equity for RL cash-balance and for converting returns → dollar equity curves. |
| **SEED** | `0` | MarginTrader, clustering | Fixes RNG for reproducibility. |
| **DEVICE** | `cuda:0` if available | MarginTrader (training) | Ensures RL model trains on GPU; classical models are CPU-bound. |

*The first three constants are fixed across every run to keep the comparison apples-to-apples; the last three apply only where relevant.*

## 6 Empirical Results and Interpretation  

**Figure 1 – Cumulative PnL Comparison** (saved as `utils/equity_curves.png`) and **Table 1** below summarise the out-of-sample performance of the four portfolio-construction paradigms under identical experimental constants (LOOKBACK = 252, EVAL_WINDOW = 5, TRANSACTION_COST = 0.002, INITIAL_CAPITAL = 1 000 000 USD, SEED = 0).

| Model                         | Total Return | Annual Return | Annual Volatility | Sharpe | Sortino |
|-------------------------------|-------------:|--------------:|------------------:|-------:|--------:|
| **GMVP (Clustering)**         | –35 %        | –4.1 %        | 14.5 % | –0.29 | –0.35 |
| **Naïve Markowitz**           | 145 %        |  8.9 %        |  6.7 % | **1.34** | **8.72** |
| **Portfolio Policy Network**  | 537 %        | 19.4 %        | 31.7 % | 0.61 | 0.72 |
| **MarginTrader (A2C)**        | –1027 %      | –98.3 %       | 212.9 % | –0.46 | –0.43 |

### 6.1 Visual inspection  

* **PPN** (red) compounds steadily through ≈ 75 % of the test horizon, accelerates mid-sample, and—despite two sharp draw-downs around day 2000—finishes at roughly 5 × initial capital.  
* **Naïve Markowitz** (green) delivers monotonic growth with low variance; draw-downs never exceed 7 %, evidencing robust risk control.  
* **GMVP** (blue) oscillates around breakeven and trends slightly negative; as a pure minimum-variance allocator it fails to overcome frictional costs.  
* **MarginTrader** (orange) remains flat until ≈ day 250, then suffers a catastrophic margin event that wipes out > 100 % of equity; the curve subsequently hugs zero because, for benchmarking purposes, we keep the episode alive and continue logging **raw** (negative) equity rather than terminating.

### 6.2 Risk-adjusted ranking  

1. **Naïve Markowitz** leads on both Sharpe and Sortino, harvesting risk premia while containing tail loss.  
2. **PPN** achieves the highest absolute return but at ≈ 5 × the volatility of Naïve Markowitz, yielding only a middling Sharpe.  
3. **GMVP** underperforms a risk-free benchmark, confirming that variance minimisation alone is insufficient in the presence of trading costs.  
4. **MarginTrader** is an extreme outlier; leverage-induced tail risk dominates its distribution, resulting in a strongly negative Sharpe.

### 6.3 Diagnostics on MarginTrader’s failure  

* **Maintenance breaches**: the A2C agent repeatedly drove its equity buffer below broker thresholds, which in live trading would trigger forced liquidations.  
* **Reward shaping**: the learning objective optimised short-horizon PnL without penalising large downside tails, encouraging excess leverage.  
* **State design**: the agent observed only price tensors—no exogenous volatility or liquidity context—limiting its ability to anticipate risk.

### 6.4 Implications for practice  

* **Risk-budget alignment**: PPN suits return-seeking mandates tolerant of 30 % volatility; Naïve Markowitz is preferable for conservative allocations.  
* **Classical methods remain competitive**: a tuned mean-variance optimiser still ranks top in risk-adjusted terms.  
* **RL in margin settings needs stricter controls**: production deployment would require hard episode termination at equity ≤ 0, dynamic leverage caps, or CVaR-aware rewards.

### 6.5 Future work  

1. **Risk-aware RL objectives** (e.g. Sharpe- or CVaR-maximisation) to penalise catastrophic paths.  
2. **Broker-realistic liquidation logic**: in production, terminate an episode immediately upon margin breach and restart from cash; for benchmarking we kept episodes alive to record a complete raw-return series.  
3. **Hybrid ensembles**: blend PPN signals with Markowitz allocation to exploit nonlinear forecasts while preserving risk control.

These findings highlight that sophisticated deep-RL architectures do not guarantee superior performance without stringent draw-down constraints, and that transparent optimisation baselines remain formidable benchmarks in academic and applied portfolio research.