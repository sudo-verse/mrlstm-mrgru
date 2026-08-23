# Model Implementation and Verification Guide

This guide provides a comprehensive overview of the implementation, directory structure, and verification procedures for the **MR-LSTM** and **MR-GRU** models. It is designed to allow professors, examiners, or peer researchers to quickly locate the model logic, understand the architecture details, and execute tests to verify implementation correctness.

---

## 1. Architectural Implementation Details

The core of this project is extending the Market-Rule-Informed Neural Network (**MRINN**) with recurrent sequence encoders. The implementation enforces a strict single-factor ablation boundary: **the structural market rules are imported unchanged from the peer checkout (`library_imbalance.model`)**, and only the history encoders are modified.

### One-Step-Ahead Model
* **Logic Location:** `library_mrrnn/model.py`
* **Description:** Instead of flattening lag sequences and feeding them to Dense layers, each of the 17 market signals gets its own recurrent cell (LSTM or GRU) to process the sequence over a real time axis (oldest to newest).
* **Parameter Cost:** Parameter size is fixed in sequence length ($T$): **4,615 parameters** for MR-GRU, **5,511 parameters** for MR-LSTM.

### Day-Ahead Multi-Horizon Model
* **Logic Location:** `library_mrrnn/multihorizon.py`
* **Description:** Forecasts all 96 quarter-hourly intervals ($t+1 \dots t+96$) of the next 24 hours in a single shot.
* **Key Components:**
  1. **`HorizonEmbedding`:** A custom Keras layer providing a learned parameter table $(96, \text{dim})$ tiled across batches. This ensures the model distinguishes between forecast horizons (preventing horizon-blindness) while remaining under the 8,900 MLP parameter limit.
  2. **Batch Folding/Unfolding:** Reshapes signals from $(N, 96, H)$ to $(N \cdot 96, H)$ before applying the rule blocks. This forces the rule head weights to be shared identically across all 96 horizons.
  3. **`DAOutageDropout`:** Simulates day-ahead publishing outages ($P_{DA} = 0$) at training time to build robustness against feed failures.

---

## 2. Directory Map (Where Code is Located)

```
MRLSTM-MRGRU/
├── library_mrrnn/                  <-- Core Model Definitions
│   ├── model.py                    <-- 1-Step Recurrent Model Architecture
│   ├── multihorizon.py             <-- Day-Ahead Multi-Horizon Model & Custom Layers
│   └── seqdata.py                  <-- Sequence Reshaping & Shared-Lag Scaler logic
├── verify.py                       <-- One-Step verification test script
├── verify_multihorizon.py          <-- Day-Ahead verification test script
├── compare.py                      <-- CLI script comparing models side-by-side
├── experiments.py                  <-- CLI script running seeds & sequence-length sweeps
├── setup.sh                        <-- One-shot environment initializer script
├── requirements.txt                <-- Pinned dependencies (TensorFlow 2.16.2, etc.)
├── notebooks/                      <-- Jupyter and Kaggle notebook scripts
│   ├── MRLSTM_Compare_Kaggle.ipynb
│   ├── MRLSTM_DayAhead_Rolling_Kaggle.ipynb
│   └── MRLSTM_DayAhead_AsinhScaler_Kaggle.ipynb
├── results/                        <-- Evaluation metric CSV tables
└── Figure/                         <-- Thesis SVG and PNG charts
```

---

## 3. Commands to Test and Verify the Models

We have provided two dedicated verification scripts that test the models end-to-end (building, compiling, fitting on dummy subsets, and performing checks).

First, initialize the environment:
```bash
cd /path/to/MRLSTM-MRGRU
bash setup.sh
```
*(This clones the external MRINN dependency, sets up a local `.venv` using Python 3.11, installs all requirements, and executes the initial smoke tests.)*

### Test 1: Verify One-Step Model
Run this command to test the 15-minute-ahead model:
```bash
.venv/bin/python verify.py
```
**What this verifies:**
1. That the local `external/MRINN` dependency rules match the original paper exactly (unmodified).
2. That inputs line up with Keras input tensors correctly (preventing silent feature permutation).
3. That the sequence time-axis runs oldest-to-newest.
4. That parameter counts match analytical expectations (GRU = 4,615; LSTM = 5,511).
5. That model checkpointing and loading operate without errors.
6. That the quantile outputs do not cross ($\text{AQCR} \equiv 0.00\%$).

### Test 2: Verify Day-Ahead Model
Run this command to test the multi-horizon model:
```bash
.venv/bin/python verify_multihorizon.py
```
**What this verifies:**
1. Correct alignment of sequence histories, cyclical calendar steps, and future day-ahead values.
2. Parameter budget matches limits (MR-GRU multi-horizon = 5,055 params).
3. `HorizonEmbedding` functions correctly and creates distinct forecast profiles across horizons (verifies standard deviation of outputs $> 0.05$).
4. The 1-year rolling training window and expanding training window folds generate the correct start/end date logic.
5. That `DAOutageDropout` successfully modifies the output forecast when day-ahead pricing is blacked out.

---

## 4. Full Guide to Reproduce Experimental Results

To replicate the results shown in the thesis:

### A. One-Step Seed Replication (Phase A)
Trains the models across 5 random seeds to verify that accuracy gains are not initialization luck.
```bash
.venv/bin/python experiments.py --phase A --epochs 50
```

### B. One-Step Sequence-Length Sweep (Phase B)
Sweeps sequence lengths $T \in \{1, 4, 16, 32, 64, 96\}$ to prove that recurrent encoders scale efficiently compared to Dense encoders.
```bash
.venv/bin/python experiments.py --phase B --epochs 50
```

### C. Compile Metrics Report
Computes Newey-West Diebold-Mariano statistical significance tests comparing the model loss series.
```bash
.venv/bin/python experiments.py --phase report
```

### D. Generate Report Figures
Generates print-ready SVG/PNG figures (AQL/MAE bar charts, forecast fan chart, accuracy-vs-parameter efficiency scatterplot) directly into the `Figure/` folder:
```bash
.venv/bin/python make_figures.py
```
