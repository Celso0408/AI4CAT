# Closed-Loop Bayesian Optimization for Catalyst Design

This repository implements a **closed-loop Bayesian Optimization (BO)** framework for catalyst discovery. It uses Gaussian Process (GP) surrogate models and acquisition strategies to iteratively suggest new candidates and simulate their performance. The objective is to optimize a multi-KPI **Merit function** under **feasibility constraints** derived from chemical design rules.

## 📁 Project Structure

```bash
.
├── BO-Data.csv                  # Initial experimental data
├── bo_closed_loop.ipynb        # Main notebook: closed-loop BO campaign
├── bo_outputs/                 # Folder containing suggestion CSVs for each generation
└── README.md                   # This file
```

---

## 📦 Dependencies

Install the following Python packages before running the notebook:

```bash
pip install numpy pandas torch gpytorch matplotlib
```

This project uses:

* **PyTorch** and **GPyTorch** for GP modeling.
* **Pandas** for data management.
* **Matplotlib** for plotting (optional).
* **Sobol sequences** and **Dirichlet sampling** for candidate generation.

---

## 🎯 Optimization Goal

We aim to **maximize a scalar Merit function** computed from 4 key performance indicators (KPIs):

1. **Cost** \[\$/kg] (to minimize)
2. **CH₄ Selectivity** \[%] (to minimize)
3. **CO₂ Conversion** \[%] (to maximize)
4. **MeOH Selectivity** \[%] (to maximize)

The individual KPIs are normalized to \[0, 1], sign-flipped if necessary (for minimization), and then averaged:

$$
\text{Merit} = \frac{1}{n} \sum_{i=1}^{n} s_i \cdot \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}
$$

where $s_i \in \{-1, +1\}$ depending on whether the KPI is to be minimized or maximized.

---

## 🧪 Design Variables

The optimization space consists of:

* Loadings (wt%) of 6 metals: `Fe`, `In`, `Ce`, `Co`, `Cu`, `Zn`
* A promoter: `K`
* A one-hot encoded support material: `SiO₂`, `Al₂O₃`, `TiO₂`, `ZrO₂`

### Constraints

The candidate generation respects **hard constraints**:

* Maximum 3 active metals per candidate
* Total loading (metals + promoter) ∈ \[2.5, 5.0] wt%
* Promoter (K) ≤ 1.0 wt%
* Exactly one support material

These constraints are applied using a feasibility mask over Sobol-sampled vectors.

---

## 🧠 Surrogate Modeling

### 1. **Merit GP Model**

A single GP regression model is trained to predict the scalar Merit:

* **Kernel**: RBF
* **Likelihood**: Gaussian
* **Training**: Adam optimizer minimizing exact marginal log-likelihood (MLL)

### 2. **KPI Models**

Each raw KPI is also modeled with a separate GP to simulate "new experiments":

* **Transformations**:

  * Cost: $\log(1 + x)$
  * Selectivities: logit transform
  * Conversion: linear
* **Kernel**: RBF or Matérn (with ARD) based on KPI type

---

## 🚀 Acquisition Function

We use **Expected Improvement (EI)** as the acquisition criterion:

$$
\text{EI}(x) = \left( \mu(x) - f_{\text{best}} \right) \Phi(z) + \sigma(x) \phi(z)
\quad \text{with} \quad z = \frac{\mu(x) - f_{\text{best}}}{\sigma(x)}
$$

Where:

* $\mu(x)$ and $\sigma(x)$ are the GP posterior mean and std
* $\Phi$ is the standard normal CDF
* $\phi$ is the standard normal PDF
* $f_{\text{best}}$ is the current best observed Merit

We sample a large batch of candidates, filter feasible ones, and select the top-k based on EI.

---

## 🔁 Closed-Loop Workflow

Each generation in the loop proceeds as:

1. **Suggest next batch** using EI
2. **Simulate** KPI values for suggested candidates
3. **Append** new data to history
4. **Retrain** KPI models (and optionally Merit GP)

---

## 📊 Output

Each generation outputs a CSV file in `bo_outputs/` containing:

* Suggested loadings and support
* Simulated KPIs (Cost, CH₄, CO₂, MeOH)
* Generation tag
* Computed Merit

---

## 🧪 Reproducibility

To reproduce the campaign:

```python
from bo_closed_loop import CatalystBO

bo = CatalystBO("BO-Data.csv")
bo._train_kpi_models()
for g in range(1, 6):
    raw = bo.suggest_next_batch(batch_size=24)
    df = pd.DataFrame(raw, columns=bo.loading_cols)
    bo._simulate_and_append(df, gen=g)
    bo._train_kpi_models()
```

---

## 🧠 Future Work

* Add **multi-objective Pareto front analysis**
* Use **Thompson Sampling** or **UCB** as acquisition
* Integrate with real lab hardware for autonomous discovery

---

## 📜 License

This repository is open-source and distributed under the MIT License.

---

Would you like this README file exported or committed into your project directory?
