import os
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
diffcp_base_path1 = "/home/quill/diffcp/examples/results/results_20250616-183709/group_lasso/data"
diffqcp_base_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/group_lasso/data"

# --- Load residuals ---
qcp_residuals = []
cp_residuals = []

for i in range(1, 11):
    qcp_resids = np.load(os.path.join(diffqcp_base_path, f"run_{i}/qcp/lsqr_residuals.npy"))
    cp_resids = np.load(os.path.join(diffcp_base_path1, f"run_{i}/qcp/lsqr_residuals.npy"))
    qcp_residuals.append(qcp_resids)
    cp_residuals.append(cp_resids)

qcp_residuals = np.concatenate(qcp_residuals)
cp_residuals = np.concatenate(cp_residuals)

print("Lasso num qcp residuals: ", qcp_residuals.shape[0])
print("Lasso mean qcp residual: ", np.mean(qcp_residuals))
print("Lasso median qcp residual: ", np.median(qcp_residuals))
print("Lasso num cp residuals: ", cp_residuals.shape[0])
print("Lasso mean cp residual: ", np.mean(cp_residuals))
print("Lasso median cp residual: ", np.median(cp_residuals))
print("=== ===")

# --- Plot ---
# plt.figure(figsize=(8, 5))

# plt.hist(cp_residuals, bins=100, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=100, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Residual Histograms for Group Lasso")
# plt.legend()
# plt.grid(True)

plt.figure(figsize=(8, 5))

bins = np.logspace(-8, -2, 100)  # Adjust as needed based on your residual range

plt.hist(cp_residuals, bins=bins, alpha=0.5, label="diffcp", color="blue", density=True)
plt.hist(qcp_residuals, bins=bins, alpha=0.5, label="diffqcp", color="orange", density=True)

plt.xscale('log')
plt.xlabel("LSQR Residual (log scale)")
plt.ylabel("Density")
plt.title("Residual Histograms for Group Lasso")
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# --- Save to SVG (for LaTeX) ---
results_dir = os.path.join(os.path.dirname(__file__), "lsqr_plots")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "group_lasso.svg")
plt.savefig(output_path, format="svg")

# === PORTFOLIO ===

# --- Paths ---
diffcp_base_path1 = "/home/quill/diffcp/examples/results/results_20250616-183709/portfolio/data"
diffqcp_base_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/portfolio/data"

# --- Load residuals ---
qcp_residuals = []
cp_residuals = []

for i in range(1, 11):
    qcp_resids = np.load(os.path.join(diffqcp_base_path, f"run_{i}/qcp/lsqr_residuals.npy"))
    cp_resids = np.load(os.path.join(diffcp_base_path1, f"run_{i}/qcp/lsqr_residuals.npy"))
    qcp_residuals.append(qcp_resids)
    cp_residuals.append(cp_resids)

qcp_residuals = np.concatenate(qcp_residuals)
cp_residuals = np.concatenate(cp_residuals)

print("Portfolio num qcp residuals: ", qcp_residuals.shape[0])
print("Portfolio mean qcp residual: ", np.mean(qcp_residuals))
print("Portfolio median qcp residual: ", np.median(qcp_residuals))
print("Portfolio num cp residuals: ", cp_residuals.shape[0])
print("Portfolio mean cp residual: ", np.mean(cp_residuals))
print("Portfolio median cp residual: ", np.median(cp_residuals))
print("=== ===")

# --- Plot ---
# plt.figure(figsize=(8, 5))

# plt.hist(cp_residuals, bins=100, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=100, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Residual Histograms for Portfolio Optimization")
# plt.legend()
# plt.grid(True)

plt.figure(figsize=(8, 5))

bins = np.logspace(-8, -2, 100)  # Adjust as needed based on your residual range

plt.hist(cp_residuals, bins=bins, alpha=0.5, label="diffcp", color="blue", density=True)
plt.hist(qcp_residuals, bins=bins, alpha=0.5, label="diffqcp", color="orange", density=True)

plt.xscale('log')
plt.xlabel("LSQR Residual (log scale)")
plt.ylabel("Density")
plt.title("Residual Histograms for Portfolio Optimization")
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# --- Save to SVG (for LaTeX) ---
results_dir = os.path.join(os.path.dirname(__file__), "lsqr_plots")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "portfolio.svg")
plt.savefig(output_path, format="svg")

# === SDP ===

# --- Paths ---
diffcp_base_path1 = "/home/quill/diffcp/examples/results/results_20250616-183709/sdp/data"
diffqcp_base_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/sdp/data"

# --- Load residuals ---
qcp_residuals = []
cp_residuals = []

for i in range(1, 11):
    qcp_resids = np.load(os.path.join(diffqcp_base_path, f"run_{i}/qcp/lsqr_residuals.npy"))
    cp_resids = np.load(os.path.join(diffcp_base_path1, f"run_{i}/qcp/lsqr_residuals.npy"))
    qcp_residuals.append(qcp_resids)
    cp_residuals.append(cp_resids)

qcp_residuals = np.concatenate(qcp_residuals)
cp_residuals = np.concatenate(cp_residuals)

print("SDP num qcp residuals: ", qcp_residuals.shape[0])
print("SDP mean qcp residual: ", np.mean(qcp_residuals))
print("SDP median qcp residual: ", np.median(qcp_residuals))
print("SDP num cp residuals: ", cp_residuals.shape[0])
print("SDP mean cp residual: ", np.mean(cp_residuals))
print("SDP median cp residual: ", np.median(cp_residuals))
print("QCP residuals min/max:", qcp_residuals.min(), qcp_residuals.max())
print("CP residuals min/max:", cp_residuals.min(), cp_residuals.max())
print("Any NaNs?", np.isnan(qcp_residuals).any(), np.isnan(cp_residuals).any())
print("Any <= 0?", (qcp_residuals <= 0).sum(), (cp_residuals <= 0).sum())
print("=== ===")

# --- Plot ---
# plt.figure(figsize=(8, 5))

# plt.hist(cp_residuals, bins=100, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=100, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Residual Histograms for SDP")
# plt.legend()
# plt.grid(True)

# plt.figure(figsize=(8, 5))

# bins = np.logspace(-8, -1, 100)  # Adjust as needed based on your residual range

# plt.hist(cp_residuals, bins=bins, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=bins, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xscale('log')
# plt.xlabel("LSQR Residual (log scale)")
# plt.ylabel("Density")
# plt.title("Residual Histograms for SDP with quadratic objective")
# plt.legend()
# plt.grid(True, which='both', ls='--', lw=0.5)

# qcp_clean = qcp_residuals[np.isfinite(qcp_residuals) & (qcp_residuals > 0)]
# cp_clean = cp_residuals[np.isfinite(cp_residuals) & (cp_residuals > 0)]

# bins = np.linspace(0, max(qcp_clean.max(), cp_clean.max()), 100)

# plt.figure(figsize=(8, 6))
# plt.hist(qcp_clean, bins=bins, alpha=0.5, label='diffqcp', color='blue', density=True)
# plt.hist(cp_clean, bins=bins, alpha=0.5, label='diffcp', color='orange', density=True)
# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Histogram of LSQR Residuals (SDP)")
# plt.legend()
# plt.tight_layout()

min_val = min(qcp_residuals.min(), cp_residuals.min())
max_val = max(qcp_residuals.max(), cp_residuals.max())
bins = np.linspace(min_val, max_val, 100)

# Plot
plt.figure(figsize=(8, 6))
plt.hist(cp_residuals, bins=bins, alpha=0.5, label='diffcp', color='blue', density=True)
plt.hist(qcp_residuals, bins=bins, alpha=0.5, label='diffqcp', color='orange', density=True)
plt.xlabel("LSQR Residual")
plt.ylabel("Density")
plt.title("Histogram of LSQR Residuals (SDP)")
plt.legend()
plt.tight_layout()

# --- Save to SVG (for LaTeX) ---
results_dir = os.path.join(os.path.dirname(__file__), "lsqr_plots")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "sdp.svg")
plt.savefig(output_path, format="svg")

# === MVDR ===

# --- Paths ---
diffcp_base_path2 = "/home/quill/diffcp/examples/results/results_20250616-194039/robust_mvdr/data"
diffqcp_base_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/robust_mvdr/data"

# --- Load residuals ---
qcp_residuals = []
cp_residuals = []

for i in range(1, 11):
    qcp_resids = np.load(os.path.join(diffqcp_base_path, f"run_{i}/qcp/lsqr_residuals.npy"))
    cp_resids = np.load(os.path.join(diffcp_base_path2, f"run_{i}/qcp/lsqr_residuals.npy"))
    qcp_residuals.append(qcp_resids)
    cp_residuals.append(cp_resids)

qcp_residuals = np.concatenate(qcp_residuals)
cp_residuals = np.concatenate(cp_residuals)

print("MVDR num qcp residuals: ", qcp_residuals.shape[0])
print("MVDR mean qcp residual: ", np.mean(qcp_residuals))
print("MVDR median qcp residual: ", np.median(qcp_residuals))
print("MVDR num cp residuals: ", cp_residuals.shape[0])
print("MVDR mean cp residual: ", np.mean(cp_residuals))
print("MVDR median cp residual: ", np.median(cp_residuals))
print("=== ===")

# --- Plot ---
# plt.figure(figsize=(8, 5))

# plt.hist(cp_residuals, bins=100, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=100, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Residual Histograms for MVDR")
# plt.legend()
# plt.grid(True)

plt.figure(figsize=(8, 5))

bins = np.logspace(-8, -2, 100)  # Adjust as needed based on your residual range

plt.hist(cp_residuals, bins=bins, alpha=0.5, label="diffcp", color="blue", density=True)
plt.hist(qcp_residuals, bins=bins, alpha=0.5, label="diffqcp", color="orange", density=True)

plt.xscale('log')
plt.xlabel("LSQR Residual (log scale)")
plt.ylabel("Density")
plt.title("Residual Histograms for MVDR")
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# --- Save to SVG (for LaTeX) ---
results_dir = os.path.join(os.path.dirname(__file__), "lsqr_plots")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "mvdr.svg")
plt.savefig(output_path, format="svg")

# === LS EQ ===

# --- Paths ---
diffcp_base_path3 = "/home/quill/diffcp/examples/results/results_20250617-191743/ls_eq_small/data"
# diffcp_base_path3 = "/home/quill/diffcp/examples/results/results_20250617-191743/ls_eq_small/data"
diffqcp_base_path = "/home/quill/diffqcp/experiments/results/results_20250617-165045/LS_eq_small_LR/data"

# --- Load residuals ---
qcp_residuals = []
cp_residuals = []

for i in range(1, 11):
    qcp_resids = np.load(os.path.join(diffqcp_base_path, f"run_{i}/qcp/lsqr_residuals.npy"))
    cp_resids = np.load(os.path.join(diffcp_base_path3, f"run_{i}/qcp/lsqr_residuals.npy"))
    qcp_residuals.append(qcp_resids)
    cp_residuals.append(cp_resids)

qcp_residuals = np.concatenate(qcp_residuals)
cp_residuals = np.concatenate(cp_residuals)

print("LS EQ num qcp residuals: ", qcp_residuals.shape[0])
print("LS EQ mean qcp residual: ", np.mean(qcp_residuals))
print("LS EQ median qcp residual: ", np.median(qcp_residuals))
print("LS EQ num cp residuals: ", cp_residuals.shape[0])
print("LS EQ mean cp residual: ", np.mean(cp_residuals))
print("LS EQ median cp residual: ", np.median(cp_residuals))
print("QCP residuals min/max:", qcp_residuals.min(), qcp_residuals.max())
print("CP residuals min/max:", cp_residuals.min(), cp_residuals.max())
print("Any NaNs?", np.isnan(qcp_residuals).any(), np.isnan(cp_residuals).any())
print("Any <= 0?", (qcp_residuals <= 0).sum(), (cp_residuals <= 0).sum())

# --- Plot ---
# plt.figure(figsize=(8, 5))

# plt.hist(cp_residuals, bins=100, alpha=0.5, label="diffcp", color="blue", density=True)
# plt.hist(qcp_residuals, bins=100, alpha=0.5, label="diffqcp", color="orange", density=True)

# plt.xlabel("LSQR Residual")
# plt.ylabel("Density")
# plt.title("Residual Histograms for MVDR")
# plt.legend()
# plt.grid(True)

plt.figure(figsize=(8, 5))

bins = np.logspace(-8, -2, 100)  # Adjust as needed based on your residual range

plt.hist(cp_residuals, bins=bins, alpha=0.5, label="diffcp", color="blue", density=True)
plt.hist(qcp_residuals, bins=bins, alpha=0.5, label="diffqcp", color="orange", density=True)

plt.xscale('log')
plt.xlabel("LSQR Residual")
plt.ylabel("Density")
plt.title("Residual Histograms for LS EQ")
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# --- Save to SVG (for LaTeX) ---
results_dir = os.path.join(os.path.dirname(__file__), "lsqr_plots")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "ls_eq.svg")
plt.savefig(output_path, format="svg")