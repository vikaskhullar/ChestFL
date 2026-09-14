# ChestFL

Federated Learning simulation for chest X-ray / medical imaging classification, with support for both **IID** and **Non-IID (NIID)** data distributions across clients.

> ⚠️ **Note:** This README was generated from the repository's current structure. Some implementation details (datasets, models, dependencies) may need to be adjusted to match the actual code.

---

## Overview

ChestFL explores Federated Learning (FL) in the context of chest imaging. It provides simulation scripts to train a shared model across multiple clients **without centralizing data**, comparing performance under:

- **IID** — data is independently and identically distributed across clients
- **NIID** — data distribution is heterogeneous (non-IID) across clients, which is more realistic in medical settings

The core goal is to study how data heterogeneity affects model convergence, accuracy, and communication efficiency in a privacy-preserving, decentralized training setup.

---

## Repository Structure

```
ChestFL/
├── Results/                        # Output logs, metrics, plots from experiments
├── simulationNewFL3 - IID.py       # FL simulation with IID client data
├── simulationNewFL3 - NIID.py      # FL simulation with non-IID client data
└── .gitattributes
```

---

## Methodology

This section explains the code logic and the federated learning pipeline implemented in both simulation scripts.

### 1. Data Preparation & Partitioning

The dataset (chest X-ray images with labels) is loaded and split across `N` simulated clients to mimic a decentralized hospital network.

**IID setting (`simulationNewFL3 - IID.py`):**
- The dataset is shuffled and split uniformly across clients.
- Each client receives approximately the same number of samples and a similar class distribution.
- This represents an ideal case where data across hospitals is statistically identical.

**Non-IID setting (`simulationNewFL3 - NIID.py`):**
- The dataset is partitioned unevenly, often by:
  - Assigning each client only a subset of classes (label skew), or
  - Using a Dirichlet distribution to control heterogeneity (quantity skew).
- This simulates real-world scenarios where different hospitals see different patient populations, equipment, or disease prevalence.

### 2. Model Architecture

- A neural network (e.g., CNN) is defined as the **global model**.
- The same architecture is replicated on each client as the **local model**.
- The model is initialized once on the server and broadcast to all clients at the start of training.

### 3. Federated Learning Loop

The training follows the standard **Federated Averaging (FedAvg)** procedure:

For each **communication round** `t = 1, 2, ..., T`:

1. **Broadcast:** The server sends the current global model weights `w_t` to all participating clients.
2. **Local Training:** Each client `k`:
   - Copies the global weights into its local model.
   - Trains on its private data for `E` local epochs using SGD/Adam.
   - Produces updated local weights `w_t^k`.
3. **Upload:** Clients send their updated weights (not data) back to the server.
4. **Aggregation:** The server combines client updates using a weighted average:
   
   ```
   w_{t+1} = Σ (n_k / n) * w_t^k
   ```
   
   where `n_k` is the number of samples on client `k` and `n` is the total number of samples.
5. **Repeat** until the maximum number of rounds is reached or convergence criteria are met.

### 4. Local Training Details

Each client’s local update typically involves:
- Forward pass through the CNN on its local batch.
- Loss computation (e.g., cross-entropy for classification).
- Backpropagation and optimizer step.
- Optional learning rate scheduling per round.

### 5. Evaluation

After each round (or at the end of training), the global model is evaluated on a **held-out test set** that was never used during training. Metrics collected include:

- Test accuracy
- Test loss
- Per-class precision / recall / F1 (if applicable)
- Communication cost (rounds to reach target accuracy)

Results from both IID and NIID runs are saved to the `Results/` directory for comparison.

### 6. Key Differences Between the Two Scripts

| Aspect | IID Script | NIID Script |
|---|---|---|
| Data split | Uniform, shuffled | Skewed (label or Dirichlet) |
| Class balance per client | Balanced | Imbalanced |
| Convergence | Faster, smoother | Slower, more volatile |
| Realism | Idealized | Realistic (clinical setting) |
| Purpose | Baseline | Robustness test |

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, Pandas
- PyTorch or TensorFlow (depending on the model implementation in the scripts)
- Matplotlib / Seaborn (for result visualization)

Install typical dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision
```

### Running a Simulation

IID setup:

```bash
python "simulationNewFL3 - IID.py"
```

Non-IID setup:

```bash
python "simulationNewFL3 - NIID.py"
```

Results (accuracy, loss curves, metrics) are saved to the `Results/` directory.

---

## Results

See the `Results/` folder for logs and performance comparisons between IID and NIID configurations. NIID training typically shows:

- Slower convergence
- Lower final accuracy
- Higher variance across clients

These effects highlight the **client drift** problem in federated learning, where local optima diverge due to heterogeneous data.

---

## Citation

If you use this code, please cite the repository:

```bibtex
@misc{khullar_chestfl,
  title  = {ChestFL: Federated Learning for Chest Imaging},
  author = {Vikash Khullar},
  year   = {2026},
  url    = {https://github.com/vikaskhullar/ChestFL}
}
```

---

## License

No license file is currently included. Please contact the repository owner before reuse.

---

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

