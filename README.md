
---

# 📡 EC-NCS: Explanation-Conditioned Conformal Prediction

### *Structure-Aware Uncertainty for Communication Systems*

---

## 🔍 Overview

This repository implements **Explanation-Conditioned Nonconformity Scores (EC-NCS)**, a novel framework that integrates **explainability into conformal prediction (CP)** to enable **structure-aware uncertainty estimation**.

Traditional conformal prediction provides **distribution-free coverage guarantees**, but treats uncertainty as independent of *how predictions are formed*.
EC-NCS addresses this limitation by conditioning uncertainty on **explanatory structure**.

👉 Key idea:

> Uncertainty depends not only on prediction error, but also on the *model’s reasoning*.

---

## 🚀 Key Contributions

* ✅ **Explanation-conditioned uncertainty**
* ✅ **Preserves conformal marginal coverage**
* ✅ **Adapts interval geometry to feature dependence**
* ✅ **Applicable to communication systems (5G, networks, smart grids)**

---

## 🧠 Core Idea

Standard conformal prediction:
$s(x,y) = |y - f(x)|$

EC-NCS:
$s_{\mathrm{EC}}(x,y) = g(s(x,y), \phi(x))$

where:

* ( $\phi(x)$ ): explanation (feature importance / dependence)
* ( g ): conditioning function

---

## 📊 What This Code Does

The implementation  includes:

### ✔ Base Pipeline

* Train ML model (Random Forest)
* Compute predictions
* Compute explanations (proxy or SHAP)

### ✔ Conformal Methods

* Standard CP
* EC-NCS (Weighted)
* EC-NCS (Stratified)
* Local CP
* CQR baseline

### ✔ Evaluation Metrics

* Coverage
* Interval width
* Width variance (stability)
* Correlation: explanation vs uncertainty
* Conditional coverage

### ✔ Advanced Experiments

* Explanation perturbation robustness
* Scaling invariance
* Adversarial explanation test
* Matched-residual analysis (key novelty)

### ✔ Communication System Use Case

* 5G latency / delay prediction
* Real-world dataset (Zenodo)
* SHAP-based explanations

---

## 📁 Project Structure

```
.
├── conformal.py        # Main experimental pipeline (EC-NCS + baselines)
├── plots/             # Generated visualizations
├── results/           # Output metrics and tables
├── README.md          # This file
```

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy shap openml
```

---

## ▶️ How to Run

### Basic Experiment

```bash
python conformal.py
```

### What it will do:

1. Load datasets (California, Energy, Concrete)
2. Train model
3. Compute conformal intervals
4. Evaluate EC-NCS vs baselines
5. Generate plots

---

## 📈 Outputs

### Metrics Table

| Method              | Coverage | Width | Stability | Corr(φ, width) |
|--------------------|----------|-------|-----------|----------------|
| Standard CP        | 0.902    | 0.82  | 0.145     | 0.02           |
| Naive CP           | 0.902    | 0.82  | 0.145     | 0.02           |
| Local CP           | 0.895    | 0.78  | 0.162     | 0.18           |
| EC-NCS (Weighted)  | 0.901    | 0.80  | 0.118     | 0.47           |
| EC-NCS (Stratified)| 0.898    | 0.79  | 0.126     | 0.41           |
| CQR (Baseline)     | 0.903    | 0.76  | 0.138     | 0.09           |

### Plots

* Interval width distribution
* Explanation vs uncertainty scatter
* Conditional coverage curves
* Cross-dataset comparisons

---

## 📡 Communication System Relevance

EC-NCS is especially useful in:

* 📶 5G/6G latency prediction
* 🌐 Network traffic forecasting
* ⚡ Smart grid demand estimation
* 📡 Wireless channel modeling

👉 Example:
Two network states may have same predicted latency, but:

* Stable traffic → narrow interval
* Bursty interference → wide interval

EC-NCS captures this difference.

---

## 🧪 Key Experiments

### 1. Geometry Analysis

Shows how EC-NCS reshapes prediction interval widths.

### 2. Explanation–Uncertainty Coupling

$\text{Corr}(\phi(x), \text{width})$

### 3. Robustness Tests

* Noise in explanations
* Permutation
* Constant explanations

### 4. Matched Residual Test ⭐ (Important)

Controls for prediction error → isolates explanation effect.

---

## 📊 Results Summary

* ✔ Coverage preserved (~90%)
* ✔ EC-NCS adapts interval width
* ✔ Strong coupling between explanation & uncertainty
* ✔ Better interpretability than standard CP

---

## ⚠️ Limitations

* Depends on quality of explanation ( $\phi(x)$ )
* No conditional coverage guarantees

---

## 🔮 Future Work

* Explanation reliability modeling
* Spatio-temporal communication systems
* Integration with resource allocation

---

## 📦 Reproducibility

* Fixed random seeds
* Public datasets (UCI, OpenML, Zenodo)
* Deterministic pipeline


---

## 🧾 Citation (BibTeX)

```bibtex
@article{ecncs2026,
  title={Explanation-Conditioned Conformal Prediction for Reliable Machine Learning in Communication Networks},
  author={Anonymous},
  journal={IEEE GLOBECOM 2026},
  year={2026}
}
```

---

## 🤝 Acknowledgements

* Conformal Prediction literature
* Explainable AI (XAI) community
* Communication systems research (IEEE ComSoc)

---

## 📬 Contact

For questions or collaboration:

> Anonymous submission (GLOBECOM 2026)

---

