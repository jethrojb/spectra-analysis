"""
Core UV-Vis Spectral Analysis Module (`uvvis_spectra.py`).

Supports:
- Auto-detecting and parsing dual UV-Vis spectrophotometer CSV export formats:
  1. Full Report format (with METHOD and SAMPLES metadata headers + leading label column)
  2. Table-Only Export format (sample names on row 1, Wavelength/Abs pairs on row 2)
- Polars DataFrame storage and spectral manipulations:
  - Sample renaming and subsetting
  - Baseline zeroing at a reference wavelength (e.g., 490 nm or 500 nm)
  - Difference spectra computation (Sample A - Sample B) and storage
  - Clean CSV export via Polars
- Peak (maxima) and trough (minima) detection using `scipy.signal.find_peaks`
- Protein concentration calculation via Beer-Lambert law:
  - Single-wavelength or dual-wavelength baseline-corrected (e.g., (A_450 - A_490) / epsilon)
  - Works on raw spectra or difference spectra
- Dual-style plotting:
  1. Minimal publication style (Matplotlib: no grid, despined top/right, vector PDF/SVG or high-DPI PNG)
  2. Interactive exploratory style (Plotly: hover tooltips, toggleable traces, marked peaks & troughs, HTML export)
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from scipy.signal import find_peaks


# Publication color palette (colorblind-friendly Okabe-Ito inspired scientific palette)
PUBLICATION_PALETTE: list[str] = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#008080",  # Teal
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#8c564b",  # Brown
    "#17becf",  # Cyan
]


@dataclass
class ConcentrationResult:
    """Stores the result and intermediate parameters of a protein concentration calculation."""

    spectrum_name: str
    target_wavelength_nm: float
    target_absorbance: float
    reference_wavelength_nm: float | None
    reference_absorbance: float
    delta_absorbance: float
    extinction_coeff: float
    extinction_coeff_unit: Literal["M^-1 cm^-1", "mM^-1 cm^-1"]
    pathlength_cm: float
    dilution_factor: float
    concentration_molar: float
    concentration: float
    unit: Literal["uM", "mM", "M"]

    def summary(self) -> str:
        """Returns a human-readable summary of the concentration calculation."""
        if self.reference_wavelength_nm is not None:
            abs_expr = (
                f"A({self.target_wavelength_nm:.1f} nm) - A({self.reference_wavelength_nm:.1f} nm) "
                f"= {self.target_absorbance:.5f} - ({self.reference_absorbance:.5f}) "
                f"= {self.delta_absorbance:.5f}"
            )
        else:
            abs_expr = (
                f"A({self.target_wavelength_nm:.1f} nm) = {self.target_absorbance:.5f}"
            )

        unit_label = "µM" if self.unit == "uM" else self.unit
        return (
            f"Spectrum          : {self.spectrum_name}\n"
            f"Absorbance Term   : {abs_expr}\n"
            f"Extinction Coeff  : {self.extinction_coeff:g} {self.extinction_coeff_unit}\n"
            f"Pathlength        : {self.pathlength_cm:g} cm\n"
            f"Dilution Factor   : {self.dilution_factor:g}x\n"
            f"Concentration     : {self.concentration:.3f} {unit_label}"
        )

    def to_latex(
        self,
        is_difference: bool = False,
        target_window: tuple[float, float] | None = None,
    ) -> str:
        """
        Generates a multi-line LaTeX equation string showing both the symbolic formula
        (adapted to the selected options) and the substituted numerical values.
        """
        abs_sym = r"\Delta A" if is_difference else "A"
        t_wl = f"{self.target_wavelength_nm:g}"

        if target_window is not None:
            w_min, w_max = min(target_window), max(target_window)
            target_sym = (
                rf"{abs_sym}_{{\text{{peak}} [{w_min:g}\text{{--}}{w_max:g}\text{{ nm}}]"
                rf" \, ({t_wl}\text{{ nm}})}}"
            )
        else:
            target_sym = rf"{abs_sym}_{{{t_wl}\text{{ nm}}}}"

        if self.reference_wavelength_nm is not None:
            r_wl = f"{self.reference_wavelength_nm:g}"
            ref_sym = rf"{abs_sym}_{{{r_wl}\text{{ nm}}}}"
            num_sym = rf"{target_sym} - {ref_sym}"
            ref_val_str = (
                f"({self.reference_absorbance:.5f})"
                if self.reference_absorbance < 0
                else f"{self.reference_absorbance:.5f}"
            )
            num_val = f"{self.target_absorbance:.5f} - {ref_val_str}"
        else:
            num_sym = target_sym
            num_val = f"{self.target_absorbance:.5f}"

        # Extinction coefficient term
        if self.extinction_coeff_unit == "mM^-1 cm^-1":
            eps_sym = r"(\varepsilon \times 10^{3})"
            eps_val = (
                rf"\left({self.extinction_coeff:,.4g}\ \text{{mM}}^{{-1}}\text{{cm}}^{{-1}}"
                rf" \times 10^{{3}}\right)"
            )
        else:
            eps_sym = r"\varepsilon"
            eps_val = rf"\left({self.extinction_coeff:,.6g}\ \text{{M}}^{{-1}}\text{{cm}}^{{-1}}\right)"

        # Unit conversion factor from M to target unit
        unit_latex = r"\mu\text{M}" if self.unit == "uM" else rf"\text{{{self.unit}}}"
        if self.unit == "uM":
            scale_sym = r" \times 10^{6}"
            scale_val = r" \times 10^{6}\ \mu\text{M}\cdot\text{M}^{-1}"
        elif self.unit == "mM":
            scale_sym = r" \times 10^{3}"
            scale_val = r" \times 10^{3}\ \text{mM}\cdot\text{M}^{-1}"
        else:
            scale_sym = ""
            scale_val = ""

        df_sym = r"\text{DF} \times " if self.dilution_factor != 1.0 else ""
        df_val = f"{self.dilution_factor:g} \\times " if self.dilution_factor != 1.0 else ""
        conc_str = (
            f"{self.concentration:.2f}"
            if abs(self.concentration) >= 1.0
            else f"{self.concentration:.4g}"
        )

        return (
            r"\begin{aligned}" + "\n"
            rf"C &= {df_sym}\frac{{{num_sym}}}{{{eps_sym} \cdot l}}{scale_sym} \\[8pt]" + "\n"
            rf"&= {df_val}\frac{{{num_val}}}{{{eps_val} \cdot \left({self.pathlength_cm:g}\ \text{{cm}}\right)}}{scale_val} "
            rf"= \mathbf{{{conc_str}\ {unit_latex}}}" + "\n"
            r"\end{aligned}"
        )


@dataclass
class SpectraDataset:
    """
    Container for UV-Vis spectra backed by a Polars DataFrame.

    The DataFrame always contains a `"wavelength"` (Float64) column and one Float64
    column per sample or computed difference spectrum.
    """

    df: pl.DataFrame
    source_name: str = "spectra"
    format_type: Literal["full_report", "table_export", "processed"] = "table_export"
    method_metadata: dict[str, str] = field(default_factory=dict)
    sample_metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    difference_columns: list[str] = field(default_factory=list)

    @property
    def wavelengths(self) -> np.ndarray:
        """Returns the wavelength column as a 1D NumPy array."""
        return self.df["wavelength"].to_numpy()

    @property
    def sample_names(self) -> list[str]:
        """Returns all spectrum column names (excluding 'wavelength')."""
        return [col for col in self.df.columns if col != "wavelength"]

    @property
    def raw_sample_names(self) -> list[str]:
        """Returns only the non-difference spectrum column names."""
        return [col for col in self.sample_names if col not in self.difference_columns]

    def copy(self) -> SpectraDataset:
        """Creates a deep copy of the SpectraDataset."""
        return SpectraDataset(
            df=self.df.clone(),
            source_name=self.source_name,
            format_type=self.format_type,
            method_metadata=dict(self.method_metadata),
            sample_metadata={k: dict(v) for k, v in self.sample_metadata.items()},
            difference_columns=list(self.difference_columns),
        )

    def rename_samples(
        self, mapping: dict[str, str], in_place: bool = True
    ) -> SpectraDataset:
        """
        Renames one or more sample columns.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old sample names to new sample names.
        in_place : bool, default True
            Whether to modify this dataset in place or return a new copy.
        """
        target = self if in_place else self.copy()
        valid_map = {
            old: new
            for old, new in mapping.items()
            if old in target.df.columns and old != "wavelength" and new.strip()
        }
        if not valid_map:
            return target

        target.df = target.df.rename(valid_map)
        for old, new in valid_map.items():
            if old in target.sample_metadata:
                target.sample_metadata[new] = target.sample_metadata.pop(old)
            if old in target.difference_columns:
                idx = target.difference_columns.index(old)
                target.difference_columns[idx] = new
        return target

    def get_absorbance_at(
        self, sample: str, wavelength_nm: float
    ) -> tuple[float, float]:
        """
        Finds the closest wavelength in the dataset to `wavelength_nm` and returns
        `(actual_wavelength_nm, absorbance)`.
        """
        if sample not in self.df.columns:
            raise KeyError(
                f"Sample '{sample}' not found. Available: {self.sample_names}"
            )

        sub = (
            self.df.select(["wavelength", sample])
            .drop_nulls()
            .with_columns(
                (pl.col("wavelength") - float(wavelength_nm)).abs().alias("_diff")
            )
            .sort("_diff")
        )
        if sub.is_empty():
            raise ValueError(f"Sample '{sample}' has no valid data points.")

        row = sub.row(0, named=True)
        return float(row["wavelength"]), float(row[sample])

    def zero_spectra(
        self,
        reference_wl: float = 490.0,
        samples: Sequence[str] | None = None,
        in_place: bool = False,
    ) -> SpectraDataset:
        """
        Baseline-corrects (zeroes) spectra by subtracting each spectrum's absorbance
        at `reference_wl` (e.g., 490 nm or 500 nm).

        Parameters
        ----------
        reference_wl : float, default 490.0
            The reference wavelength (nm) at which absorbance will be set to 0.
        samples : Sequence[str] | None, default None
            Specific sample columns to zero. If None, zeroes all sample columns.
        in_place : bool, default False
            If True, updates this dataset in-place; otherwise returns a new dataset.
        """
        target = self if in_place else self.copy()
        cols_to_zero = list(samples) if samples is not None else target.sample_names

        exprs: list[pl.Expr] = []
        for col in cols_to_zero:
            if col not in target.df.columns or col == "wavelength":
                raise KeyError(
                    f"Sample '{col}' not found. Available: {target.sample_names}"
                )
            _, ref_abs = target.get_absorbance_at(col, reference_wl)
            exprs.append((pl.col(col) - ref_abs).alias(col))

        if exprs:
            target.df = target.df.with_columns(exprs)
        return target

    def compute_difference(
        self,
        minuend: str,
        subtrahend: str,
        name: str | None = None,
        zero_at: float | None = None,
    ) -> str:
        """
        Computes `minuend - subtrahend` (i.e. Sample 1 - Sample 2), stores it as a column
        in the Polars DataFrame, and returns the column name.

        Parameters
        ----------
        minuend : str
            Name of the first spectrum (e.g., 'CO' or 'Sample 3').
        subtrahend : str
            Name of the spectrum to subtract (e.g., 'Fe(II)' or 'Sample 2').
        name : str | None, default None
            Custom column name for the difference spectrum. Defaults to `f"{minuend} - {subtrahend}"`.
        zero_at : float | None, default None
            Optional wavelength (nm) at which to baseline-zero the resulting difference spectrum.
        """
        for s in (minuend, subtrahend):
            if s not in self.df.columns:
                raise KeyError(
                    f"Spectrum '{s}' not found. Available: {self.sample_names}"
                )

        diff_name = name.strip() if (name and name.strip()) else f"{minuend} - {subtrahend}"
        self.df = self.df.with_columns(
            (pl.col(minuend) - pl.col(subtrahend)).alias(diff_name)
        )

        if zero_at is not None:
            _, ref_val = self.get_absorbance_at(diff_name, zero_at)
            self.df = self.df.with_columns(
                (pl.col(diff_name) - ref_val).alias(diff_name)
            )

        if diff_name not in self.difference_columns:
            self.difference_columns.append(diff_name)

        return diff_name

    def detect_extrema(
        self,
        samples: Sequence[str] | None = None,
        prominence: float = 0.005,
        distance: int = 3,
        wavelength_range: tuple[float, float] | None = None,
        find_positive_peaks: bool = True,
        find_negative_troughs: bool = True,
    ) -> pl.DataFrame:
        """
        Detects positive peaks (local maxima) and/or negative troughs (local minima)
        across the specified spectra using `scipy.signal.find_peaks`.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with columns:
            `["sample", "type", "wavelength", "absorbance", "prominence"]`
        """
        target_samples = list(samples) if samples is not None else self.sample_names
        records: list[dict[str, Any]] = []

        # Sort by ascending wavelength for consistent peak detection
        sorted_df = self.df.sort("wavelength")
        if wavelength_range is not None:
            wl_min, wl_max = min(wavelength_range), max(wavelength_range)
            sorted_df = sorted_df.filter(
                (pl.col("wavelength") >= wl_min) & (pl.col("wavelength") <= wl_max)
            )

        wls = sorted_df["wavelength"].to_numpy()

        for sample in target_samples:
            if sample not in sorted_df.columns or sample == "wavelength":
                continue
            y = sorted_df[sample].to_numpy()
            valid_mask = ~np.isnan(y)
            valid_wls = wls[valid_mask]
            valid_y = y[valid_mask]

            if len(valid_y) < 3:
                continue

            if find_positive_peaks:
                peak_indices, props = find_peaks(
                    valid_y, prominence=prominence, distance=distance
                )
                prominences = props.get("prominences", np.zeros(len(peak_indices)))
                for idx, prom in zip(peak_indices, prominences):
                    records.append(
                        {
                            "sample": sample,
                            "type": "peak",
                            "wavelength": float(valid_wls[idx]),
                            "absorbance": float(valid_y[idx]),
                            "prominence": float(prom),
                        }
                    )

            if find_negative_troughs:
                trough_indices, props = find_peaks(
                    -valid_y, prominence=prominence, distance=distance
                )
                prominences = props.get("prominences", np.zeros(len(trough_indices)))
                for idx, prom in zip(trough_indices, prominences):
                    records.append(
                        {
                            "sample": sample,
                            "type": "trough",
                            "wavelength": float(valid_wls[idx]),
                            "absorbance": float(valid_y[idx]),
                            "prominence": float(prom),
                        }
                    )

        schema = {
            "sample": pl.String,
            "type": pl.String,
            "wavelength": pl.Float64,
            "absorbance": pl.Float64,
            "prominence": pl.Float64,
        }
        if not records:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(records, schema=schema).sort(
            ["sample", "type", "wavelength"]
        )

    def calculate_concentration(
        self,
        spectrum: str,
        extinction_coeff: float = 91000.0,
        target_wl: float | tuple[float, float] = 450.0,
        reference_wl: float | None = 490.0,
        dilution_factor: float = 1.0,
        pathlength_cm: float = 1.0,
        extinction_coeff_unit: Literal["M^-1 cm^-1", "mM^-1 cm^-1"] = "M^-1 cm^-1",
        unit: Literal["uM", "mM", "M"] = "uM",
    ) -> ConcentrationResult:
        """
        Calculates protein concentration from a raw or difference spectrum.

        By default, uses the dual-wavelength equation typical for P450 CO-difference spectra:
            Concentration = dilution_factor * (A_target - A_ref) / (extinction_coeff * pathlength_cm)
        If `reference_wl=None`, uses the single-wavelength Beer-Lambert form:
            Concentration = dilution_factor * A_target / (extinction_coeff * pathlength_cm)

        Parameters
        ----------
        spectrum : str
            Column name of the sample or difference spectrum to evaluate.
        extinction_coeff : float, default 91000.0
            Extinction coefficient value (e.g., 91000 M^-1 cm^-1 or 91 mM^-1 cm^-1).
        target_wl : float | tuple[float, float], default 450.0
            Either a single target wavelength (nm), or a `(min_wl, max_wl)` window tuple
            (e.g., `(445.0, 455.0)`) within which the maximum absorbance peak is automatically selected.
        reference_wl : float | None, default 490.0
            Reference wavelength (nm) for baseline subtraction, or `None` for no subtraction.
        dilution_factor : float, default 1.0
            Dilution multiplier (e.g., 50.0 for a 1:50 dilution).
        pathlength_cm : float, default 1.0
            Optical pathlength in cm.
        extinction_coeff_unit : {"M^-1 cm^-1", "mM^-1 cm^-1"}, default "M^-1 cm^-1"
            Units of the supplied extinction coefficient.
        unit : {"uM", "mM", "M"}, default "uM"
            Desired output concentration unit.
        """
        if spectrum not in self.df.columns or spectrum == "wavelength":
            raise KeyError(
                f"Spectrum '{spectrum}' not found. Available: {self.sample_names}"
            )
        if extinction_coeff <= 0:
            raise ValueError("extinction_coeff must be positive.")
        if pathlength_cm <= 0:
            raise ValueError("pathlength_cm must be positive.")

        if isinstance(target_wl, (tuple, list)) and len(target_wl) == 2:
            w_min, w_max = min(target_wl), max(target_wl)
            window_df = (
                self.df.select(["wavelength", spectrum])
                .drop_nulls()
                .filter(
                    (pl.col("wavelength") >= float(w_min))
                    & (pl.col("wavelength") <= float(w_max))
                )
                .sort(spectrum, descending=True)
            )
            if window_df.is_empty():
                raise ValueError(
                    f"No wavelength points found in window [{w_min}, {w_max}] nm."
                )
            best_row = window_df.row(0, named=True)
            actual_target_wl = float(best_row["wavelength"])
            target_abs = float(best_row[spectrum])
        else:
            actual_target_wl, target_abs = self.get_absorbance_at(
                spectrum, float(target_wl)  # type: ignore[arg-type]
            )

        if reference_wl is not None:
            actual_ref_wl, ref_abs = self.get_absorbance_at(
                spectrum, float(reference_wl)
            )
            delta_abs = target_abs - ref_abs
        else:
            actual_ref_wl = None
            ref_abs = 0.0
            delta_abs = target_abs

        # Convert extinction coefficient to M^-1 cm^-1
        eps_molar = (
            extinction_coeff * 1000.0
            if extinction_coeff_unit == "mM^-1 cm^-1"
            else extinction_coeff
        )

        conc_molar = (
            dilution_factor * delta_abs / (eps_molar * pathlength_cm)
        )

        scale_map: dict[str, float] = {"M": 1.0, "mM": 1e3, "uM": 1e6}
        if unit not in scale_map:
            raise ValueError(f"Unsupported unit '{unit}'. Choose from 'uM', 'mM', 'M'.")

        conc_scaled = conc_molar * scale_map[unit]

        return ConcentrationResult(
            spectrum_name=spectrum,
            target_wavelength_nm=actual_target_wl,
            target_absorbance=target_abs,
            reference_wavelength_nm=actual_ref_wl,
            reference_absorbance=ref_abs,
            delta_absorbance=delta_abs,
            extinction_coeff=extinction_coeff,
            extinction_coeff_unit=extinction_coeff_unit,
            pathlength_cm=pathlength_cm,
            dilution_factor=dilution_factor,
            concentration_molar=conc_molar,
            concentration=conc_scaled,
            unit=unit,
        )

    def export_csv(
        self,
        output_path: str | Path,
        samples: Sequence[str] | None = None,
    ) -> Path:
        """
        Exports the Polars DataFrame (or a subset of sample columns) to a clean CSV file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if samples is not None:
            cols = ["wavelength"] + [
                s for s in samples if s in self.df.columns and s != "wavelength"
            ]
            self.df.select(cols).write_csv(out)
        else:
            self.df.write_csv(out)
        return out


def _deduplicate_names(names: Sequence[str]) -> list[str]:
    """Ensures all sample names are non-empty and unique."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for idx, raw_name in enumerate(names):
        base = raw_name.strip() if raw_name and raw_name.strip() else f"Sample {idx + 1}"
        if base == "wavelength":
            base = "Sample_wavelength"
        if base not in seen:
            seen[base] = 1
            unique.append(base)
        else:
            seen[base] += 1
            unique.append(f"{base} ({seen[base]})")
    return unique


def load_spectra(
    source: str | Path | bytes | IO[str] | IO[bytes],
    source_name: str | None = None,
) -> SpectraDataset:
    """
    Auto-detects and parses UV-Vis spectrophotometer CSV files in either:
      1. Full Report Format (e.g., `2026-11-09-mb2xstar.csv`)
      2. Table-Only Export Format (e.g., `Export Data 2026-09-18-H6.csv`)
      3. Previously exported clean Polars CSV (containing a `wavelength` column)

    Parameters
    ----------
    source : str | Path | bytes | IO
        Filepath or file-like buffer containing the CSV data.
    source_name : str | None, default None
        Optional display name when reading from an in-memory buffer.

    Returns
    -------
    SpectraDataset
        A `SpectraDataset` holding the parsed `polars.DataFrame` and metadata.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        resolved_name = source_name or path.stem
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig", errors="replace")
        resolved_name = source_name or "uploaded_spectra"
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            text = str(raw)
        resolved_name = source_name or getattr(source, "name", "uploaded_spectra")
        resolved_name = Path(resolved_name).stem
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV file is empty.")

    # Check if this is already a clean Polars-exported CSV with a "wavelength" column
    first_non_empty = next((r for r in rows if any(c.strip() for c in r)), None)
    if first_non_empty and first_non_empty[0].strip().lower() == "wavelength":
        df = pl.read_csv(io.StringIO(text)).with_columns(
            pl.col(c).cast(pl.Float64) for c in pl.read_csv(io.StringIO(text)).columns
        )
        first_col = df.columns[0]
        if first_col != "wavelength":
            df = df.rename({first_col: "wavelength"})
        return SpectraDataset(
            df=df,
            source_name=resolved_name,
            format_type="processed",
        )

    # Locate the header row containing "Wavelength (nm)" and "Abs"
    header_row_idx: int | None = None
    wl_col_indices: list[int] = []

    for r_idx, row in enumerate(rows):
        normalized = [cell.strip().lower() for cell in row]
        if any(c.startswith("wavelength") for c in normalized):
            header_row_idx = r_idx
            wl_col_indices = [
                c_idx
                for c_idx, cell in enumerate(normalized)
                if cell.startswith("wavelength")
            ]
            break

    if header_row_idx is None or not wl_col_indices:
        raise ValueError(
            "Could not locate a 'Wavelength (nm)' header row in the CSV file."
        )

    # Determine format type based on header position and preceding rows
    first_wl_col = wl_col_indices[0]
    format_type: Literal["full_report", "table_export"] = (
        "full_report" if (first_wl_col >= 1 or header_row_idx > 2) else "table_export"
    )

    method_metadata: dict[str, str] = {}
    sample_metadata: dict[str, dict[str, str]] = {}
    raw_sample_names: list[str] = []

    if format_type == "full_report":
        section = "HEADER"
        sample_meta_rows: list[list[str]] = []
        name_row: list[str] | None = None

        for r_idx in range(header_row_idx):
            row = rows[r_idx]
            non_empty = [c.strip() for c in row if c.strip()]
            if not non_empty:
                continue
            first_cell = row[0].strip()
            if first_cell.upper() == "METHOD" and len(non_empty) == 1:
                section = "METHOD"
                continue
            if first_cell.upper() == "SAMPLES" and len(non_empty) == 1:
                section = "SAMPLES"
                continue

            if section in ("HEADER", "METHOD"):
                if len(row) >= 2 and first_cell:
                    method_metadata[first_cell] = row[1].strip()
            elif section == "SAMPLES":
                if first_cell.lower() == "name":
                    name_row = row
                else:
                    sample_meta_rows.append(row)

        if name_row is None and header_row_idx >= 1:
            name_row = rows[header_row_idx - 1]

        for wl_col in wl_col_indices:
            s_name = (
                name_row[wl_col].strip()
                if (name_row is not None and wl_col < len(name_row))
                else ""
            )
            raw_sample_names.append(s_name)

        sample_names = _deduplicate_names(raw_sample_names)

        for s_name, wl_col in zip(sample_names, wl_col_indices):
            s_meta: dict[str, str] = {}
            for m_row in sample_meta_rows:
                key = m_row[0].strip()
                if key and wl_col < len(m_row) and m_row[wl_col].strip():
                    s_meta[key] = m_row[wl_col].strip()
            sample_metadata[s_name] = s_meta

    else:
        # Table-only export format: sample names are on the row immediately above header_row_idx
        name_row = rows[header_row_idx - 1] if header_row_idx >= 1 else []
        for wl_col in wl_col_indices:
            s_name = name_row[wl_col].strip() if wl_col < len(name_row) else ""
            raw_sample_names.append(s_name)
        sample_names = _deduplicate_names(raw_sample_names)

    # Parse numeric data rows after header_row_idx
    sample_dfs: list[pl.DataFrame] = []
    for s_name, wl_col in zip(sample_names, wl_col_indices):
        abs_col = wl_col + 1
        wls: list[float] = []
        abs_vals: list[float] = []

        for row in rows[header_row_idx + 1 :]:
            if abs_col >= len(row):
                continue
            wl_str = row[wl_col].strip()
            abs_str = row[abs_col].strip()
            if not wl_str or not abs_str:
                continue
            try:
                wls.append(float(wl_str))
                abs_vals.append(float(abs_str))
            except ValueError:
                continue

        if wls:
            s_df = pl.DataFrame(
                {
                    "wavelength": pl.Series("wavelength", wls, dtype=pl.Float64),
                    s_name: pl.Series(s_name, abs_vals, dtype=pl.Float64),
                }
            )
            sample_dfs.append(s_df)

    if not sample_dfs:
        raise ValueError("No numeric wavelength/absorbance data found in CSV.")

    # Combine sample DataFrames on wavelength
    combined_df = sample_dfs[0]
    for s_df in sample_dfs[1:]:
        if combined_df["wavelength"].equals(s_df["wavelength"]):
            combined_df = combined_df.with_columns(s_df[s_df.columns[1]])
        else:
            combined_df = combined_df.join(
                s_df, on="wavelength", how="full", coalesce=True
            )

    return SpectraDataset(
        df=combined_df,
        source_name=resolved_name,
        format_type=format_type,
        method_metadata=method_metadata,
        sample_metadata=sample_metadata,
    )


def plot_publication(
    dataset: SpectraDataset,
    samples: Sequence[str] | None = None,
    difference_samples: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Absorbance",
    diff_ylabel: str = r"$\Delta$ Absorbance",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    diff_ylim: tuple[float, float] | None = None,
    mark_peaks: bool = False,
    mark_troughs: bool = False,
    peak_prominence: float = 0.005,
    show_zero_line: bool = True,
    linewidth: float = 1.8,
    figsize: tuple[float, float] | None = None,
    palette: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """
    Creates a minimal, publication-ready Matplotlib figure (no gridlines, despined top/right).

    Supports:
    - Single-panel mode: when only `samples` (or only `difference_samples`) are provided.
    - Two-panel stacked mode: when both `samples` (top panel) and `difference_samples` (bottom panel)
      are provided.
    """
    colors = list(palette) if palette else PUBLICATION_PALETTE

    if samples is None and difference_samples is None:
        raw_cols = dataset.raw_sample_names
        diff_cols = dataset.difference_columns
        if diff_cols and raw_cols:
            samples = raw_cols
            difference_samples = diff_cols
        else:
            samples = dataset.sample_names
            difference_samples = None

    has_top = bool(samples)
    has_bottom = bool(difference_samples)
    two_panel = has_top and has_bottom

    if figsize is None:
        figsize = (6.5, 6.8) if two_panel else (6.5, 4.2)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.1,
        }
    )

    if two_panel:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
        )
        axes_pairs = [
            (ax_top, list(samples or []), ylabel, ylim, False),
            (ax_bot, list(difference_samples or []), diff_ylabel, diff_ylim, True),
        ]
    else:
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        active_samples = list(samples if has_top else (difference_samples or []))
        is_diff_only = has_bottom and not has_top
        active_ylabel = diff_ylabel if is_diff_only else ylabel
        active_ylim = diff_ylim if is_diff_only else ylim
        axes_pairs = [
            (ax_single, active_samples, active_ylabel, active_ylim, is_diff_only)
        ]

    sorted_df = dataset.df.sort("wavelength")
    wls = sorted_df["wavelength"].to_numpy()

    for ax, panel_samples, panel_ylabel, panel_ylim, is_diff_panel in axes_pairs:
        # Minimal scientific aesthetic: no grid, remove top and right spines
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            direction="out",
            length=4.5,
            width=1.1,
            colors="#111111",
            labelsize=10,
        )

        # Show y=0 reference line on difference panel or if values cross zero
        panel_min = 0.0
        for s in panel_samples:
            if s in sorted_df.columns:
                arr = sorted_df[s].to_numpy()
                if len(arr) > 0 and not np.all(np.isnan(arr)):
                    panel_min = min(panel_min, float(np.nanmin(arr)))

        if show_zero_line and (is_diff_panel or s in dataset.difference_columns or panel_min < -0.002):
            ax.axhline(0, color="#333333", linestyle=":", linewidth=1.0, zorder=1)

        for idx, sample in enumerate(panel_samples):
            if sample not in sorted_df.columns:
                continue
            color = colors[idx % len(colors)]
            y = sorted_df[sample].to_numpy()
            ax.plot(wls, y, label=sample, color=color, linewidth=linewidth, zorder=2)

            if mark_peaks or mark_troughs:
                extrema_df = dataset.detect_extrema(
                    samples=[sample],
                    prominence=peak_prominence,
                    wavelength_range=xlim,
                    find_positive_peaks=mark_peaks,
                    find_negative_troughs=mark_troughs,
                )
                for row in extrema_df.iter_rows(named=True):
                    marker = "^" if row["type"] == "peak" else "v"
                    va = "bottom" if row["type"] == "peak" else "top"
                    offset_y = 6 if row["type"] == "peak" else -8
                    ax.scatter(
                        [row["wavelength"]],
                        [row["absorbance"]],
                        color=color,
                        marker=marker,
                        s=38,
                        zorder=4,
                    )
                    ax.annotate(
                        f"{row['wavelength']:.0f} nm",
                        xy=(row["wavelength"], row["absorbance"]),
                        xytext=(0, offset_y),
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=8.5,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_ylabel(panel_ylabel, fontsize=11, color="#111111")
        if panel_ylim is not None:
            ax.set_ylim(panel_ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(float(np.nanmin(wls)), float(np.nanmax(wls)))

        if panel_samples:
            ax.legend(
                frameon=False,
                fontsize=9.5,
                loc="best",
            )

    # Bottom axis gets the wavelength label
    axes_pairs[-1][0].set_xlabel(
        xlabel, fontsize=11, color="#111111"
    )

    if title:
        axes_pairs[0][0].set_title(
            title, fontsize=12, fontweight="bold", pad=10, color="#111111"
        )

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=False)

    return fig


def plot_interactive(
    dataset: SpectraDataset,
    samples: Sequence[str] | None = None,
    difference_samples: Sequence[str] | None = None,
    title: str | None = None,
    mark_peaks: bool = True,
    mark_troughs: bool = True,
    peak_prominence: float = 0.005,
    wavelength_range: tuple[float, float] | None = None,
    show_zero_line: bool = True,
    output_html: str | Path | None = None,
) -> go.Figure:
    """
    Creates an interactive Plotly figure with hover inspection, clickable legend traces,
    and annotated markers for both positive peaks and negative troughs.

    Can optionally save the interactive figure as a standalone `.html` file.
    """
    if samples is None and difference_samples is None:
        raw_cols = dataset.raw_sample_names
        diff_cols = dataset.difference_columns
        if diff_cols and raw_cols:
            samples = raw_cols
            difference_samples = diff_cols
        else:
            samples = dataset.sample_names
            difference_samples = None

    has_top = bool(samples)
    has_bottom = bool(difference_samples)
    two_panel = has_top and has_bottom

    if two_panel:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Raw / Selected Spectra", "Difference Spectra"),
            row_heights=[0.55, 0.45],
        )
        panels = [
            (1, list(samples or []), "Absorbance", False),
            (2, list(difference_samples or []), "Δ Absorbance", True),
        ]
    else:
        fig = make_subplots(rows=1, cols=1)
        active_samples = list(samples if has_top else (difference_samples or []))
        is_diff_only = has_bottom and not has_top
        panels = [
            (
                1,
                active_samples,
                "Δ Absorbance" if is_diff_only else "Absorbance",
                is_diff_only,
            )
        ]

    sorted_df = dataset.df.sort("wavelength")
    if wavelength_range is not None:
        w_min, w_max = min(wavelength_range), max(wavelength_range)
        sorted_df = sorted_df.filter(
            (pl.col("wavelength") >= w_min) & (pl.col("wavelength") <= w_max)
        )

    wls = sorted_df["wavelength"].to_numpy()

    for row_idx, panel_samples, y_title, is_diff_panel in panels:
        if show_zero_line and is_diff_panel:
            fig.add_hline(
                y=0.0,
                line_dash="dot",
                line_color="#555555",
                line_width=1.2,
                row=row_idx,
                col=1,
            )

        for idx, sample in enumerate(panel_samples):
            if sample not in sorted_df.columns:
                continue
            color = PUBLICATION_PALETTE[idx % len(PUBLICATION_PALETTE)]
            y = sorted_df[sample].to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=wls,
                    y=y,
                    mode="lines",
                    name=sample,
                    legendgroup=sample,
                    line=dict(color=color, width=2.3),
                    hovertemplate=(
                        f"<b>{sample}</b><br>"
                        "Wavelength: %{x:.1f} nm<br>"
                        "Absorbance: %{y:.5f}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

            if mark_peaks or mark_troughs:
                extrema_df = dataset.detect_extrema(
                    samples=[sample],
                    prominence=peak_prominence,
                    wavelength_range=wavelength_range,
                    find_positive_peaks=mark_peaks,
                    find_negative_troughs=mark_troughs,
                )

                if not extrema_df.is_empty():
                    peaks_sub = extrema_df.filter(pl.col("type") == "peak")
                    if not peaks_sub.is_empty():
                        fig.add_trace(
                            go.Scatter(
                                x=peaks_sub["wavelength"].to_list(),
                                y=peaks_sub["absorbance"].to_list(),
                                mode="markers+text",
                                name=f"{sample} (Peaks)",
                                legendgroup=sample,
                                showlegend=False,
                                marker=dict(
                                    symbol="triangle-up",
                                    size=11,
                                    color=color,
                                    line=dict(width=1, color="#ffffff"),
                                ),
                                text=[
                                    f"{w:.0f} nm ({a:.3f})"
                                    for w, a in zip(
                                        peaks_sub["wavelength"].to_list(),
                                        peaks_sub["absorbance"].to_list(),
                                    )
                                ],
                                textposition="top center",
                                textfont=dict(size=10, color=color),
                                hovertemplate=(
                                    f"<b>{sample} — Peak (λ_max)</b><br>"
                                    "Wavelength: %{x:.1f} nm<br>"
                                    "Absorbance: %{y:.5f}<extra></extra>"
                                ),
                            ),
                            row=row_idx,
                            col=1,
                        )

                    troughs_sub = extrema_df.filter(pl.col("type") == "trough")
                    if not troughs_sub.is_empty():
                        fig.add_trace(
                            go.Scatter(
                                x=troughs_sub["wavelength"].to_list(),
                                y=troughs_sub["absorbance"].to_list(),
                                mode="markers+text",
                                name=f"{sample} (Troughs)",
                                legendgroup=sample,
                                showlegend=False,
                                marker=dict(
                                    symbol="triangle-down",
                                    size=11,
                                    color=color,
                                    line=dict(width=1, color="#ffffff"),
                                ),
                                text=[
                                    f"{w:.0f} nm ({a:.3f})"
                                    for w, a in zip(
                                        troughs_sub["wavelength"].to_list(),
                                        troughs_sub["absorbance"].to_list(),
                                    )
                                ],
                                textposition="bottom center",
                                textfont=dict(size=10, color=color),
                                hovertemplate=(
                                    f"<b>{sample} — Trough (λ_min)</b><br>"
                                    "Wavelength: %{x:.1f} nm<br>"
                                    "Absorbance: %{y:.5f}<extra></extra>"
                                ),
                            ),
                            row=row_idx,
                            col=1,
                        )

        fig.update_yaxes(
            title_text=y_title,
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.35)",
            zeroline=False,
            row=row_idx,
            col=1,
        )

    fig.update_xaxes(
        title_text="Wavelength (nm)",
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.35)",
        row=2 if two_panel else 1,
        col=1,
    )

    chart_title = title or f"UV-Vis Spectra — {dataset.source_name}"
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=18)),
        template="plotly_white",
        hovermode="closest",
        height=760 if two_panel else 520,
        legend=dict(
            title=dict(text="Spectra (click to toggle)"),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=65, r=180, t=70, b=60),
    )

    if output_html is not None:
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn")

    return fig
