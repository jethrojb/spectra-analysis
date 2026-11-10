"""
Interactive Cell Workflow (`workflow_interactive.py`).

Run individual `# %%` cells interactively in VS Code / Antigravity (using the `env` kernel)
or execute the entire script from the command line to test the full workflow on both CSV formats.
"""

# %% [markdown]
# # UV-Vis Spectral Analysis — Interactive Workflow
# **Step 1**: Load CSV into a Polars-backed `SpectraDataset` & inspect all samples interactively.
# **Step 2**: Rename samples, optionally zero at a reference wavelength, and compute difference spectra.
# **Step 3**: Calculate protein concentration using adjustable Beer-Lambert parameters.
# **Step 4**: Export minimal publication figures (`.pdf` / `.png`) and processed Polars `.csv` files.

# %% Step 1: Load & Inspect Spectra (Both CSV Export Formats)
import matplotlib.pyplot as plt
import polars as pl
from uvvis_spectra import load_spectra, plot_interactive, plot_publication

# Load Format 1 (Full Report CSV) and Format 2 (Table-Only Export CSV)
ds_mb2 = load_spectra("2026-11-09-mb2xstar.csv")
ds_h6 = load_spectra("Export Data 2026-09-18-H6.csv")

print(f"Loaded {ds_mb2.source_name} ({ds_mb2.format_type}): {ds_mb2.sample_names}")
print(f"Loaded {ds_h6.source_name} ({ds_h6.format_type}): {ds_h6.sample_names}")

# Inspect Polars DataFrame head
print("\nPolars DataFrame Preview (Export Data 2026-09-18-H6):")
print(ds_h6.df.head(5))

# Detect positive peaks and negative troughs across all samples in Export Data 2026-09-18-H6
extrema_h6 = ds_h6.detect_extrema(prominence=0.005)
print("\nDetected Peaks & Troughs:")
print(extrema_h6)

# Save interactive HTML plot for exploratory sample identification
fig_int_explore = plot_interactive(
    ds_h6,
    title="Exploratory Inspection — Export Data 2026-09-18-H6",
    mark_peaks=True,
    mark_troughs=True,
    output_html="outputs/H6_step1_inspect_interactive.html",
)

# %% Step 2: Zero Baselines & Compute Difference Spectra (Sample A - Sample B)
# Example A: On Export Data 2026-09-18-H6.csv
# Sample 5 has a strong Soret peak at 447 nm (CO-bound), Sample 2 is Fe(II)
diff_h6_name = ds_h6.compute_difference(
    minuend="Sample 5",
    subtrahend="Sample 2",
    name="Sample 5 - Sample 2 (CO - II)",
    zero_at=None,  # Or set e.g. zero_at=490.0 to zero the difference spectrum at 490 nm
)

# Example B: On 2026-11-09-mb2xstar.csv
# Optional baseline zeroing of raw spectra at 490 nm before or during analysis
ds_mb2_zeroed = ds_mb2.zero_spectra(reference_wl=490.0)
diff_mb2_name = ds_mb2.compute_difference(
    minuend="Sample 3",
    subtrahend="Sample 1",
    name="Sample 3 - Sample 1",
)

# Export updated Polars DataFrame (including the stored difference spectrum) to CSV
ds_h6.export_csv("outputs/H6_with_difference.csv")
ds_mb2.export_csv("outputs/mb2xstar_with_difference.csv")

# %% Step 3: Protein Concentration Calculation
# Standard P450 CO-difference calculation: (A_peak[445-455] - A_490) / 91000 * dilution_factor
conc_p450 = ds_h6.calculate_concentration(
    spectrum=diff_h6_name,
    extinction_coeff=91000.0,
    target_wl=(445.0, 455.0),  # Automatically picks peak at 447 nm in [445, 455] nm window
    reference_wl=490.0,
    dilution_factor=50.0,
    unit="uM",
)
print("\n=== P450 Concentration (Difference Spectrum, Dual-Wavelength) ===")
print(conc_p450.summary())

# Single-sample Soret band concentration example (e.g., oxidized H6-iii-100X at 417 nm, no ref subtraction)
conc_ferric = ds_h6.calculate_concentration(
    spectrum="H6-iii-100X",
    extinction_coeff=100000.0,
    target_wl=(412.0, 422.0),
    reference_wl=None,  # Single-wavelength A_target / epsilon
    dilution_factor=100.0,
    unit="uM",
)
print("\n=== Ferric Sample Concentration (Single Spectrum, Single-Wavelength) ===")
print(conc_ferric.summary())

# %% Step 4: Generate Publication & Interactive Plots (Raw + Difference)
# 1. Interactive HTML with both raw spectra and difference spectrum (peaks & troughs marked)
fig_int_diff = plot_interactive(
    ds_h6,
    samples=["H6-iii-100X", "Sample 2", "Sample 5"],
    difference_samples=[diff_h6_name],
    title="H6 UV-Vis & CO-Difference Spectrum",
    mark_peaks=True,
    mark_troughs=True,
    output_html="outputs/H6_step2_difference_interactive.html",
)

# 2. Minimal Publication Plot (Two-panel: Raw on top, Difference on bottom, no grid, despined)
fig_pub_two = plot_publication(
    ds_h6,
    samples=["H6-iii-100X", "Sample 2", "Sample 5"],
    difference_samples=[diff_h6_name],
    mark_peaks=True,
    mark_troughs=False,
    output_path="outputs/H6_publication_two_panel.png",
    dpi=300,
)
fig_pub_two.savefig("outputs/H6_publication_two_panel.pdf", bbox_inches="tight")
plt.close(fig_pub_two)

# 3. Minimal Publication Plot for 2026-11-09-mb2xstar.csv
fig_pub_mb2 = plot_publication(
    ds_mb2,
    samples=["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
    difference_samples=[diff_mb2_name],
    output_path="outputs/mb2xstar_publication.png",
    dpi=300,
)
plt.close(fig_pub_mb2)
print("\nAll interactive HTML, publication PNG/PDF, and Polars CSV files written to ./outputs/")
