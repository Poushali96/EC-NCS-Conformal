
# EC-NCS: Explanation-Conditioned Conformal Prediction
## Validated Gains and Failure Boundaries in Communication-Network Delay

This repository provides experimental implementations,
configurations, and supporting results for the manuscript:

**Explanation-Conditioned Conformal Prediction for
Communication-Network Delay: Validated Gains and Failure Boundaries**

The study investigates whether explanations from an
existing network-delay predictor can improve conformal
prediction intervals through validation-gated
explanation-dependent scaling.


## 1. Overview

Accurate network-delay predictions are important for communication-network monitoring and resource management. However, prediction errors can vary across radio conditions, mobility patterns, traffic conditions, and measurement environments.

Standard split conformal prediction (CP) constructs prediction intervals around an existing predictor. Its usual absolute-residual formulation assigns a constant interval radius across observations. This may be inefficient when prediction errors differ substantially across network conditions.

**Explanation-Conditioned Normalized Conformal Prediction (EC-NCS)** investigates whether model explanations from an already fitted predictor can provide useful information for adjusting prediction interval widths.

EC-NCS uses explanation-derived information to estimate relative variation in prediction errors. A validation procedure determines whether to use the adaptive scale or retain Standard CP. The selected score rule is fixed before final conformal calibration.

The objective is to improve prediction interval quality without retraining the underlying point predictor.

## 2. Research Questions

The study addresses three main questions:

1. Can explanations from an existing network-delay predictor help identify variation in prediction errors and improve conformal interval quality?
2. Under what statistical assumptions does explanation-dependent normalization preserve finite-sample marginal coverage, and when can it improve interval efficiency?
3. How reliably can adaptive scaling be estimated and selected when network measurements are heterogeneous, dependent within runs, or affected by changes in operating conditions?

The study examines both improvements and failure cases, including situations where explanation information is weak or rare prediction-error patterns are difficult to estimate.

## 3. EC-NCS Method

### Explanation representation

For a fitted predictor $\widehat f$, let $E_j(x;\widehat f)$ be the attribution assigned to feature $j$.

The total explanation magnitude is

$$
\Phi(x)=\sum_{j=1}^{p}|E_j(x;\widehat f)|.
$$

We map this quantity to a training-derived rank:

$$
Z(x)=F_{\Phi,\mathrm{tr}}(\Phi(x)).
$$

This rank is used to learn a positive scale $s(Z(x))$ from training data.

### Normalized conformal score

The EC-NCS nonconformity score is

$$
A_i=\frac{|Y_i-\widehat f(X_i)|}{s(Z(X_i))}.
$$

After calibration, the prediction interval for a new input \(x\) is

$$
\widehat C_\alpha(x)=
\left[
\widehat f(x)-\widehat q\,s(Z(x)),
\widehat f(x)+\widehat q\,s(Z(x))
\right].
$$

Here, $\widehat q$ is the conformal quantile calculated from a separate calibration sample.

### Validation-based activation

The practical EC-NCS procedure compares adaptive scaling with Standard CP using training-only validation data.

The implemented gate requires a minimum 1% improvement in the validation objective and permits a maximum coverage shortfall of 0.02 below the nominal target.

When these conditions are not satisfied, the procedure selects

$$
s(Z)\equiv 1,
$$

which returns the Standard CP interval.

The selection is completed before outer conformal calibration.

## 4. Theoretical Analysis

The manuscript examines the following statistical properties:

- **Marginal coverage:** Conditions under which explanation-dependent normalization retains the finite-sample guarantee of split conformal prediction.
- **Interval efficiency:** Conditions under which explanation information can reduce expected Winkler loss relative to a constant-radius interval.
- **Scale-estimation error:** How errors in estimating the adaptive radius can reduce the potential efficiency improvement.
- **Validation-based policy selection:** A confidence-controlled theoretical selection rule for comparing candidate policies.
- **Run-level coverage:** Coverage for a randomly selected measurement from a future run and simultaneous coverage across an entire run.
- **Covariate shift:** Coverage conditions under changes in the input distribution, including the effect of errors in importance-weight estimation.

The theoretical confidence-controlled selector is distinct from the simpler practical validation gate evaluated in the experiments.

## 5. Experimental Evaluation

### Controlled simulations

The controlled experiments evaluate explanation conditioning when its relationship with prediction errors is known.

They examine informative and uninformative conditioning, scale estimation, validation-based activation, and rare-mechanism behavior.

### WUE/NTNU network-delay experiments

The WUE/NTNU evaluations examine interval performance on real network-delay measurements.

They include within-site evaluations and cross-site transfer experiments. The latter assess empirical performance under changing measurement environments.

### CICV5G network-delay experiments

CICV5G is used to evaluate EC-NCS on heterogeneous network measurements grouped into complete driving runs.

The study includes ordinary development experiments and an independent evaluation using 20 reserved runs.

The independent evaluation separates complete training, calibration, and test runs. Each reserved run appears in the test set once across five outer folds.

### Additional experiments

The supplementary experiments include:

- Regression benchmarks on California Housing, Energy Efficiency, and Concrete Compressive Strength.
- Explanation perturbation and degradation analysis.
- Sensitivity to the number of explanation strata.
- Computational cost measurements.
- Rare-mechanism evaluation and statistical comparisons.

Some earlier experiments use weighted and stratified EC-NCS variants. These should be distinguished from the validation-gated EC-NCS procedure evaluated in the main manuscript.

## 6. Main Findings

The experiments show that the usefulness of explanation-based scaling depends on whether the explanation representation captures variation in prediction errors.

In the independent CICV5G evaluation, validation-gated EC-NCS reduced Winkler loss by **13.2%** relative to Standard CP for the Random Forest predictor. This difference remained statistically significant after Holm correction $(p_{\mathrm{Holm}}=0.001751)$.

The corresponding improvement was not statistically significant for the MLP predictor. Conformalized quantile regression also achieved better overall interval quality in the independent evaluation.

The WUE/NTNU results show that direct residual normalization can outperform explanation-based scaling.

Rare-mechanism experiments demonstrate that adaptive scaling may struggle when larger prediction errors occur infrequently. The detailed experiments examine these failure cases and compare EC-NCS with alternative uncertainty estimation methods.

Together, the results establish that explanation conditioning can improve interval quality in selected settings, but does not consistently outperform other adaptive conformal methods.

## 7. Repository Contents

The repository currently contains the following principal files.

| File | Description |
|---|---|
| `conformal.py` | Python implementation and experimental pipeline for EC-NCS variants and baselines |
| `EC_NCS_EXTRA_pynb.ipynb` | Experimental notebook containing additional EC-NCS evaluations |
| `locked_config.json` | Saved configuration for the rare-mechanism evaluation |
| `all_seed_metrics.csv` | Experimental metrics across settings and seeds |
| `gate_acceptance_summary.csv` | Summary of validation-gate decisions |
| `paired_tests_holm.csv` | Paired statistical comparisons with Holm correction |
| `prespecified_primary_tests_holm.csv` | Primary statistical comparison results |
| `bootstrap_confidence_intervals.csv` | Bootstrap confidence interval summaries |
| `anchor_method_comparison.csv` | Method comparison results |
| `integrity.json` | Saved integrity-check information |
| `ec_ncs_rare_mechanism_gated_results_archive.zip` | Archived rare-mechanism experimental results |
| `fig_*.png`, `fig_*.pdf` | Selected experimental figures |
| `LICENSE` | BSD 3-Clause license |

The repository contains experimental material developed across different stages of the study. Not every file represents the final validation-gated pipeline.

## 8. Installation

The experiments use Python and commonly available scientific computing packages.

For the baseline experimental pipeline, the dependencies include:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy shap openml
```

Additional dependencies may be required for specific notebook experiments and network datasets.

## 9. Running the Experiments

Clone the repository:

```bash
git clone https://github.com/Poushali96/EC-NCS-Conformal.git
cd EC-NCS-Conformal
```

Run the existing baseline implementation:

```bash
python conformal.py
```

This script includes the earlier regression benchmark experiments and comparisons among conformal methods.

The notebook `EC_NCS_EXTRA_pynb.ipynb` contains additional experimental implementations. Some sections use Google Colab and require dataset access, configuration, or path adjustments.

The saved CSV files and experiment archives provide supporting numerical results.

**Reproducing every result in the journal manuscript requires matching the corresponding experimental configuration, dataset preparation procedure, random seeds, and evaluation protocol.** The repository should be treated as a collection of research implementations and supporting outputs until a complete, independently verified reproduction workflow is available.

## 10. Evaluation Metrics

The experiments evaluate uncertainty intervals using:

**Empirical coverage:** Fraction of observed targets contained within their prediction intervals.

**Coverage error:** Absolute difference between empirical coverage and the nominal target.

**Mean interval width:** Average distance between the upper and lower prediction limits.

**Winkler loss:** Interval-quality score that penalizes both width and observations falling outside their intervals.

Statistical comparisons use paired observations, with Holm correction applied to the relevant hypothesis families.

For the independent CICV5G analysis, complete driving runs are used as the statistical units.

## 11. Limitations

The main limitations include dependence on the quality of the explanation representation and the accuracy of the estimated scaling function.

Marginal conformal coverage does not guarantee equal coverage across every network condition. Adaptation may also offer little benefit when explanation information does not reflect variation in prediction errors.

Rare conditions can make local residual estimation difficult. In addition, explanation generation adds computational cost, and changes between calibration and deployment conditions may affect empirical coverage.

The reported results should therefore be interpreted according to the assumptions and evaluation protocols described in the manuscript.

## 12. Citation

If you use the code or results, please cite the associated manuscript:

```bibtex

@misc{ecncs2026,
  title = {Explanation-Conditioned Conformal Prediction
           for Communication-Network Delay:
           Validated Gains and Failure Boundaries},
  author = {Anonymous},
  year = {2026},
  note = {Manuscript under anonymous review}
}

```

The citation should be updated when the manuscript receives a permanent publication or preprint identifier.

## 13. License

This repository is distributed under the BSD 3-Clause license. See `LICENSE` for details.


