"""
Command-Line Interface (`uvvis_cli.py`) for UV-Vis Spectral Analysis.

Supports a two-step terminal workflow:
  Step 1 (`inspect`): Parse a CSV, display samples & detected peaks/troughs, and save/open an interactive HTML plot.
  Step 2 (`diff`)   : Compute Sample A - Sample B, optionally zero at a reference wavelength, calculate protein
                      concentration, export processed Polars CSV, and render publication/interactive plots.
  Step 3 (`conc`)   : Calculate protein concentration directly on any sample or difference pair.

Examples
--------
1. Inspect spectra interactively:
   python uvvis_cli.py inspect "2026-11-09-mb2xstar.csv" --open

2. Compute difference spectrum, calculate P450 concentration, and export both publication PDF & interactive HTML:
   python uvvis_cli.py diff "Export Data 2026-09-18-H6.csv" -a "Sample 5" -b "Sample 2" --name "CO - Fe(II)" --style both --ext-coeff 91000 --target-wl 445 455 --ref-wl 490 --dilution 50 -o h6_analysis
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from uvvis_spectra import (
    load_spectra,
    plot_interactive,
    plot_publication,
)


def _parse_rename_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parses `OLD=NEW` strings into a dictionary."""
    mapping: dict[str, str] = {}
    if not pairs:
        return mapping
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"Invalid rename format '{item}'. Expected 'Old Name=New Name'."
            )
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def cmd_inspect(args: argparse.Namespace) -> int:
    """Handles the `inspect` subcommand."""
    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        return 1

    ds = load_spectra(csv_path)

    if args.rename:
        ds.rename_samples(_parse_rename_pairs(args.rename), in_place=True)

    if args.zero_wl is not None:
        ds.zero_spectra(reference_wl=args.zero_wl, in_place=True)

    selected_samples = args.samples if args.samples else ds.sample_names

    print(f"\n=== Dataset: {ds.source_name} (Format: {ds.format_type}) ===")
    if ds.method_metadata:
        scan_start = ds.method_metadata.get("Scan range start (nm)", "?")
        scan_stop = ds.method_metadata.get("Scan range stop (nm)", "?")
        print(f"Scan Range : {scan_start} nm -> {scan_stop} nm")
    print(f"Wavelengths: {len(ds.df)} points ({ds.df['wavelength'].min():.1f} - {ds.df['wavelength'].max():.1f} nm)")
    print(f"Samples ({len(ds.sample_names)}): {', '.join(ds.sample_names)}")

    extrema = ds.detect_extrema(
        samples=selected_samples,
        prominence=args.prominence,
        find_positive_peaks=not args.no_peaks,
        find_negative_troughs=not args.no_troughs,
    )
    print("\n--- Detected Extrema (Peaks & Troughs) ---")
    if extrema.is_empty():
        print("No peaks/troughs found with current prominence threshold.")
    else:
        with pl.Config(tbl_rows=50, tbl_cols=10, fmt_str_lengths=40):
            print(extrema)

    if args.save_csv:
        csv_out = ds.export_csv(args.save_csv, samples=selected_samples)
        print(f"\nSaved Polars CSV to: {csv_out.resolve()}")

    stem_out = Path(args.output) if args.output else Path(f"{ds.source_name}_inspect")

    if args.style in ("interactive", "both"):
        html_path = (
            stem_out
            if stem_out.suffix.lower() == ".html"
            else stem_out.with_suffix(".html")
        )
        plot_interactive(
            ds,
            samples=selected_samples,
            title=args.title or f"UV-Vis Spectra — {ds.source_name}",
            mark_peaks=not args.no_peaks,
            mark_troughs=not args.no_troughs,
            peak_prominence=args.prominence,
            output_html=html_path,
        )
        print(f"Saved interactive plot to: {html_path.resolve()}")
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())

    if args.style in ("pub", "both"):
        pub_ext = (
            stem_out.suffix
            if stem_out.suffix.lower() in (".png", ".pdf", ".svg")
            else f".{args.pub_format}"
        )
        pub_path = stem_out.with_suffix(pub_ext)
        fig = plot_publication(
            ds,
            samples=selected_samples,
            title=args.title,
            mark_peaks=args.pub_mark_peaks,
            mark_troughs=args.pub_mark_troughs,
            peak_prominence=args.prominence,
            output_path=pub_path,
            dpi=args.dpi,
        )
        plt.close(fig)
        print(f"Saved publication plot to: {pub_path.resolve()}")

    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Handles the `diff` subcommand."""
    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        return 1

    ds = load_spectra(csv_path)

    if args.rename:
        ds.rename_samples(_parse_rename_pairs(args.rename), in_place=True)

    if args.zero_wl is not None:
        ds.zero_spectra(reference_wl=args.zero_wl, in_place=True)

    diff_col = ds.compute_difference(
        minuend=args.minuend,
        subtrahend=args.subtrahend,
        name=args.name,
        zero_at=args.zero_diff_wl,
    )
    print(f"\nComputed difference spectrum: '{diff_col}' ({args.minuend} - {args.subtrahend})")

    extrema = ds.detect_extrema(
        samples=[diff_col],
        prominence=args.prominence,
        find_positive_peaks=True,
        find_negative_troughs=True,
    )
    print("\n--- Difference Spectrum Extrema ---")
    if extrema.is_empty():
        print("No peaks/troughs detected on difference spectrum.")
    else:
        with pl.Config(tbl_rows=30, tbl_cols=10):
            print(extrema)

    if args.ext_coeff is not None:
        target_wl: float | tuple[float, float]
        if len(args.target_wl) == 1:
            target_wl = float(args.target_wl[0])
        elif len(args.target_wl) == 2:
            target_wl = (float(args.target_wl[0]), float(args.target_wl[1]))
        else:
            raise ValueError("--target-wl accepts either 1 wavelength or 2 wavelengths (min max).")

        ref_wl = None if args.no_ref_wl else args.ref_wl
        conc_res = ds.calculate_concentration(
            spectrum=diff_col,
            extinction_coeff=args.ext_coeff,
            target_wl=target_wl,
            reference_wl=ref_wl,
            dilution_factor=args.dilution,
            pathlength_cm=args.pathlength,
            extinction_coeff_unit=args.ext_coeff_unit,
            unit=args.unit,
        )
        print("\n--- Protein Concentration Calculation ---")
        print(conc_res.summary())

    stem_out = (
        Path(args.output)
        if args.output
        else Path(f"{ds.source_name}_{args.minuend}_minus_{args.subtrahend}".replace(" ", "_"))
    )

    if args.save_csv:
        csv_out = Path(args.save_csv)
    else:
        csv_out = stem_out.with_suffix(".csv")
    ds.export_csv(csv_out)
    print(f"\nSaved Polars CSV (with '{diff_col}') to: {csv_out.resolve()}")

    top_samples: list[str] | None
    if args.diff_only:
        top_samples = None
    elif args.samples:
        top_samples = args.samples
    else:
        top_samples = [args.minuend, args.subtrahend]

    if args.style in ("interactive", "both"):
        html_path = (
            stem_out
            if stem_out.suffix.lower() == ".html"
            else stem_out.with_suffix(".html")
        )
        plot_interactive(
            ds,
            samples=top_samples,
            difference_samples=[diff_col],
            title=args.title or f"{ds.source_name}: {diff_col}",
            mark_peaks=True,
            mark_troughs=True,
            peak_prominence=args.prominence,
            output_html=html_path,
        )
        print(f"Saved interactive plot to: {html_path.resolve()}")
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())

    if args.style in ("pub", "both"):
        pub_ext = (
            stem_out.suffix
            if stem_out.suffix.lower() in (".png", ".pdf", ".svg")
            else f".{args.pub_format}"
        )
        pub_path = stem_out.with_suffix(pub_ext)
        fig = plot_publication(
            ds,
            samples=top_samples,
            difference_samples=[diff_col],
            title=args.title,
            mark_peaks=args.pub_mark_peaks,
            mark_troughs=args.pub_mark_troughs,
            peak_prominence=args.prominence,
            output_path=pub_path,
            dpi=args.dpi,
        )
        plt.close(fig)
        print(f"Saved publication plot to: {pub_path.resolve()}")

    return 0


def cmd_conc(args: argparse.Namespace) -> int:
    """Handles the `conc` subcommand for direct concentration calculation."""
    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        return 1

    ds = load_spectra(csv_path)

    if args.subtrahend:
        target_col = ds.compute_difference(
            minuend=args.spectrum,
            subtrahend=args.subtrahend,
        )
    else:
        target_col = args.spectrum

    if len(args.target_wl) == 1:
        target_wl: float | tuple[float, float] = float(args.target_wl[0])
    elif len(args.target_wl) == 2:
        target_wl = (float(args.target_wl[0]), float(args.target_wl[1]))
    else:
        raise ValueError("--target-wl accepts either 1 wavelength or 2 wavelengths (min max).")

    ref_wl = None if args.no_ref_wl else args.ref_wl
    res = ds.calculate_concentration(
        spectrum=target_col,
        extinction_coeff=args.ext_coeff,
        target_wl=target_wl,
        reference_wl=ref_wl,
        dilution_factor=args.dilution,
        pathlength_cm=args.pathlength,
        extinction_coeff_unit=args.ext_coeff_unit,
        unit=args.unit,
    )
    print("\n=== Protein Concentration Calculation ===")
    print(res.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvvis_cli",
        description="UV-Vis Spectrophotometer Parser, Difference Engine, Concentration Calculator & Plotter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcommand: inspect ---
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Step 1: Parse CSV, list samples & peaks/troughs, and plot spectra.",
    )
    p_inspect.add_argument("file", help="Path to spectrophotometer CSV file.")
    p_inspect.add_argument(
        "-s", "--samples", nargs="+", help="Subset of sample names to plot."
    )
    p_inspect.add_argument(
        "--rename",
        nargs="+",
        metavar="OLD=NEW",
        help="Rename sample columns, e.g. 'Sample 1=Fe(III)' 'Sample 2=Fe(II)'.",
    )
    p_inspect.add_argument(
        "--zero-wl",
        type=float,
        default=None,
        help="Optional wavelength (nm) at which to baseline-zero all spectra (e.g. 490).",
    )
    p_inspect.add_argument(
        "--style",
        choices=["interactive", "pub", "both"],
        default="interactive",
        help="Plot style to generate (default: interactive).",
    )
    p_inspect.add_argument(
        "--pub-format",
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="File format for publication plot when --style is pub or both (default: pdf).",
    )
    p_inspect.add_argument(
        "--prominence",
        type=float,
        default=0.005,
        help="Minimum peak/trough prominence for scipy.signal.find_peaks (default: 0.005).",
    )
    p_inspect.add_argument(
        "--no-peaks", action="store_true", help="Do not mark positive peaks on interactive plot."
    )
    p_inspect.add_argument(
        "--no-troughs", action="store_true", help="Do not mark negative troughs on interactive plot."
    )
    p_inspect.add_argument(
        "--pub-mark-peaks", action="store_true", help="Mark peaks on publication plot."
    )
    p_inspect.add_argument(
        "--pub-mark-troughs", action="store_true", help="Mark troughs on publication plot."
    )
    p_inspect.add_argument("--title", default=None, help="Custom plot title.")
    p_inspect.add_argument("--dpi", type=int, default=300, help="Publication plot DPI.")
    p_inspect.add_argument(
        "-o", "--output", default=None, help="Output file path or stem."
    )
    p_inspect.add_argument(
        "--save-csv", default=None, help="Export parsed Polars table to CSV path."
    )
    p_inspect.add_argument(
        "--open", action="store_true", help="Automatically open the interactive HTML in browser."
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # --- Subcommand: diff ---
    p_diff = subparsers.add_parser(
        "diff",
        help="Step 2: Compute Sample A - Sample B difference spectrum, concentration, and plots.",
    )
    p_diff.add_argument("file", help="Path to spectrophotometer CSV file.")
    p_diff.add_argument(
        "-a", "--minuend", required=True, help="First sample name (Sample A in A - B)."
    )
    p_diff.add_argument(
        "-b", "--subtrahend", required=True, help="Second sample name to subtract (Sample B in A - B)."
    )
    p_diff.add_argument(
        "--name", default=None, help="Custom name for the difference spectrum (e.g. 'CO - Fe(II)')."
    )
    p_diff.add_argument(
        "-s",
        "--samples",
        nargs="+",
        help="Raw samples to display in top panel alongside the difference panel.",
    )
    p_diff.add_argument(
        "--diff-only",
        action="store_true",
        help="Plot only the difference spectrum (single-panel plot).",
    )
    p_diff.add_argument(
        "--rename",
        nargs="+",
        metavar="OLD=NEW",
        help="Rename sample columns before computing difference.",
    )
    p_diff.add_argument(
        "--zero-wl",
        type=float,
        default=None,
        help="Zero all raw spectra at this wavelength (nm) before subtraction.",
    )
    p_diff.add_argument(
        "--zero-diff-wl",
        type=float,
        default=None,
        help="Zero the resulting difference spectrum at this wavelength (nm), e.g. 490.",
    )
    p_diff.add_argument(
        "--style",
        choices=["interactive", "pub", "both"],
        default="both",
        help="Plot style to generate (default: both).",
    )
    p_diff.add_argument(
        "--pub-format",
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="File format for publication plot (default: pdf).",
    )
    p_diff.add_argument(
        "--prominence",
        type=float,
        default=0.005,
        help="Minimum peak/trough prominence (default: 0.005).",
    )
    p_diff.add_argument(
        "--pub-mark-peaks", action="store_true", help="Mark peaks on publication plot."
    )
    p_diff.add_argument(
        "--pub-mark-troughs", action="store_true", help="Mark troughs on publication plot."
    )
    # Concentration arguments inside diff
    p_diff.add_argument(
        "--ext-coeff",
        type=float,
        default=None,
        help="Extinction coefficient to calculate concentration (e.g. 91000 for P450).",
    )
    p_diff.add_argument(
        "--ext-coeff-unit",
        choices=["M^-1 cm^-1", "mM^-1 cm^-1"],
        default="M^-1 cm^-1",
        help="Unit of extinction coefficient (default: 'M^-1 cm^-1').",
    )
    p_diff.add_argument(
        "--target-wl",
        nargs="+",
        type=float,
        default=[445.0, 455.0],
        help="Target wavelength (1 value, e.g. 450) or peak search window (2 values, e.g. 445 455).",
    )
    p_diff.add_argument(
        "--ref-wl",
        type=float,
        default=490.0,
        help="Reference wavelength for baseline subtraction in concentration calc (default: 490.0).",
    )
    p_diff.add_argument(
        "--no-ref-wl",
        action="store_true",
        help="Disable reference wavelength subtraction (use single-wavelength A_target / epsilon).",
    )
    p_diff.add_argument(
        "--dilution",
        type=float,
        default=1.0,
        help="Dilution factor multiplier (default: 1.0).",
    )
    p_diff.add_argument(
        "--pathlength",
        type=float,
        default=1.0,
        help="Cuvette pathlength in cm (default: 1.0).",
    )
    p_diff.add_argument(
        "--unit",
        choices=["uM", "mM", "M"],
        default="uM",
        help="Output concentration unit (default: uM).",
    )
    p_diff.add_argument("--title", default=None, help="Custom plot title.")
    p_diff.add_argument("--dpi", type=int, default=300, help="Publication plot DPI.")
    p_diff.add_argument(
        "-o", "--output", default=None, help="Output file stem for plots/CSV."
    )
    p_diff.add_argument(
        "--save-csv", default=None, help="Custom path for exported Polars CSV."
    )
    p_diff.add_argument(
        "--open", action="store_true", help="Open interactive HTML plot in browser."
    )
    p_diff.set_defaults(func=cmd_diff)

    # --- Subcommand: conc ---
    p_conc = subparsers.add_parser(
        "conc",
        help="Calculate protein concentration from a sample or difference spectrum.",
    )
    p_conc.add_argument("file", help="Path to spectrophotometer CSV file.")
    p_conc.add_argument(
        "-s",
        "--spectrum",
        required=True,
        help="Sample name (or minuend sample if --subtrahend is also given).",
    )
    p_conc.add_argument(
        "-b",
        "--subtrahend",
        default=None,
        help="Optional second sample to subtract before calculating concentration.",
    )
    p_conc.add_argument(
        "--ext-coeff",
        type=float,
        default=91000.0,
        help="Extinction coefficient (default: 91000.0).",
    )
    p_conc.add_argument(
        "--ext-coeff-unit",
        choices=["M^-1 cm^-1", "mM^-1 cm^-1"],
        default="M^-1 cm^-1",
        help="Extinction coefficient unit (default: 'M^-1 cm^-1').",
    )
    p_conc.add_argument(
        "--target-wl",
        nargs="+",
        type=float,
        default=[450.0],
        help="Target wavelength (e.g. 450) or peak search window (e.g. 445 455).",
    )
    p_conc.add_argument(
        "--ref-wl",
        type=float,
        default=490.0,
        help="Reference wavelength (default: 490.0). Use --no-ref-wl to disable.",
    )
    p_conc.add_argument(
        "--no-ref-wl",
        action="store_true",
        help="Use single-wavelength calculation without subtracting a reference wavelength.",
    )
    p_conc.add_argument(
        "--dilution",
        type=float,
        default=1.0,
        help="Dilution factor multiplier (default: 1.0).",
    )
    p_conc.add_argument(
        "--pathlength",
        type=float,
        default=1.0,
        help="Optical pathlength in cm (default: 1.0).",
    )
    p_conc.add_argument(
        "--unit",
        choices=["uM", "mM", "M"],
        default="uM",
        help="Output concentration unit (default: uM).",
    )
    p_conc.set_defaults(func=cmd_conc)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
