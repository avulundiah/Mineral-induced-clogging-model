#!/usr/bin/env python
# coding: utf-8

# ===============================================================================================
# 
# $$\text{Benchmark power-law and normalized Kozeny--Carman closures from Baek et al. (2019), Water Resources Research, Table 2}$$
# $$\text{Avulundiah Edwin Phiri}$$
# $$\text{Department of Mathematics, Copperbelt University, Kitwe, Zambia}$$
# $$\text{24/07/2026}$$
# 
# ==============================================================================================
# 

# In[1]:


from csv import DictWriter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


OUT = Path(__file__).resolve().parent
TIMES_H = np.array([0, 17, 27, 36, 52, 58], dtype=float)
POROSITY = np.array([31.9, 28.2, 22.1, 20.7, 17.6, 17.2]) / 100.0
K_NORMALIZED = np.array([1.0, 0.6021, 0.1602, 0.0816, 0.0012, 7.54e-13 / 1.23e-9])
PHI0 = POROSITY[0]
PHI = POROSITY[1:]
OBSERVED = K_NORMALIZED[1:]


def power_law(n):
    return (PHI / PHI0) ** n


def normalized_kozeny_carman():
    return (PHI / PHI0) ** 3 * ((1 - PHI0) / (1 - PHI)) ** 2


def rmse(predicted):
    return float(np.sqrt(np.mean((predicted - OBSERVED) ** 2)))


def log_rmse(predicted):
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(OBSERVED)) ** 2)))


def main():
    predictions = {"Power n=5": power_law(5.0),
                   "Power n=9.7": power_law(9.7),
                   "Kozeny--Carman": normalized_kozeny_carman()}
    results = []
    for name, predicted in predictions.items():
        results.append({
            "closure": name,
            "RMSE_normalized_permeability": rmse(predicted),
            "LogRMSE_natural_log": log_rmse(predicted),
            "final_predicted_k_over_k0": float(predicted[-1]),
            "final_observed_k_over_k0": float(OBSERVED[-1]),
            "final_overprediction_factor": float(predicted[-1] / OBSERVED[-1]),
        })
    with (OUT / "kozeny_carman_comparison_metrics.csv").open("w", newline="") as stream:
        writer = DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    stages = []
    for i in range(len(OBSERVED)):
        stages.append({
            "time_h": float(TIMES_H[i + 1]),
            "porosity": float(PHI[i]),
            "phi_over_phi0": float(PHI[i] / PHI0),
            "observed_k_over_k0": float(OBSERVED[i]),
            "power_n5": float(predictions["Power n=5"][i]),
            "power_n9p7": float(predictions["Power n=9.7"][i]),
            "kozeny_carman": float(predictions["Kozeny--Carman"][i]),
        })
    with (OUT / "kozeny_carman_stage_predictions.csv").open("w", newline="") as stream:
        writer = DictWriter(stream, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)

    # A single comparison plot replaces the six-curve candidate-exponent
    # figure, avoiding additional space in the main manuscript.
    relative_porosity = np.linspace(POROSITY[-1] / PHI0, 1.0, 350)
    absolute_porosity = PHI0 * relative_porosity
    fig, ax = plt.subplots(figsize=(6.0, 3.55))
    ax.plot(relative_porosity, relative_porosity ** 5,
            color="#1b5e86", lw=1.8, label=r"Power law ($n=5$)")
    ax.plot(relative_porosity, relative_porosity ** 9.7,
            color="#bf7130", lw=1.8, ls="--", label=r"Power law ($n=9.7$)")
    ax.plot(relative_porosity, relative_porosity ** 3 *
            ((1-PHI0) / (1-absolute_porosity)) ** 2,
            color="#39866a", lw=1.9, ls=":", label="Kozeny--Carman")
    ax.scatter(POROSITY / PHI0, K_NORMALIZED, color="#262626",
               s=27, marker="o", zorder=4, label="Baek et al. (2019)")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.5)
    ax.set_xlim(0.51, 1.02)
    ax.set_xlabel(r"Relative porosity, $\phi/\phi_0$")
    ax.set_ylabel(r"Relative permeability, $k/k_0$")
    ax.grid(alpha=0.19, which="major")
    ax.legend(loc="lower right", frameon=True, fontsize=8.2)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / "powerlaw_vs_kozeny_carman.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Two scale-specific optimal exponents, reported as diagnostics only;
    # these are fitted to the same five data points, not held-out validation.
    for criterion in (rmse, log_rmse):
        optimum = minimize_scalar(
            lambda n: criterion(power_law(n)),
            bounds=(0.1, 30.0), method="bounded",
        )
        print(f"Continuous optimal n for {criterion.__name__}: "
              f"{optimum.x:.6f}, error={optimum.fun:.6f}")
    for row in results:
        print(f"{row['closure']:>15}: RMSE={row['RMSE_normalized_permeability']:.6f}, "
              f"LogRMSE={row['LogRMSE_natural_log']:.6f}, "
              f"final_pred={row['final_predicted_k_over_k0']:.6f}, "
              f"final_overprediction={row['final_overprediction_factor']:.1f}x")


if __name__ == "__main__":
    main()

