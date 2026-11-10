"""
Streamlit Web Application (`streamlit_app.py`) for UV-Vis Spectral Analysis.

Launch locally with:
    .\\env\\Scripts\\streamlit.exe run streamlit_app.py
or double-click `run_app.bat`.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from uvvis_spectra import (
    SpectraDataset,
    load_spectra,
    plot_interactive,
    plot_publication,
)

st.set_page_config(
    page_title="UV-Vis Spectral Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 UV-Vis Spectral Analyzer")
st.caption(
    "Auto-parses spectrophotometer CSV exports (Full Report & Table-Only formats) with Polars • "
    "Interactive Peak/Trough Inspection • Difference Spectra • Protein Concentration • Publication Plots"
)

# --- Session State Initialization ---
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# --- Sidebar: Data Loading & Global Settings ---
with st.sidebar:
    st.header("1. Load CSV Data")
    uploaded_file = st.file_uploader(
        "Upload Spectrophotometer CSV",
        type=["csv"],
        help="Supports both Full Report CSVs (with METHOD/SAMPLES blocks) and Table-Only Export CSVs.",
    )

    local_csvs = sorted(
        [
            p.name
            for p in Path(".").glob("*.csv")
            if not p.name.endswith("_with_difference.csv")
        ]
    )
    selected_local = st.selectbox(
        "Or select a local CSV in workspace:",
        options=["(None)"] + local_csvs,
        index=1 if local_csvs else 0,
    )

    source_id: str | None = None
    raw_ds: SpectraDataset | None = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_id = f"upload:{uploaded_file.name}:{len(file_bytes)}"
        if st.session_state.current_file_id != source_id:
            try:
                raw_ds = load_spectra(
                    file_bytes, source_name=Path(uploaded_file.name).stem
                )
                st.session_state.dataset = raw_ds
                st.session_state.current_file_id = source_id
            except Exception as exc:
                st.error(f"Could not parse uploaded file: {exc}")
    elif selected_local != "(None)":
        source_id = f"local:{selected_local}"
        if st.session_state.current_file_id != source_id:
            try:
                raw_ds = load_spectra(selected_local)
                st.session_state.dataset = raw_ds
                st.session_state.current_file_id = source_id
            except Exception as exc:
                st.error(f"Could not parse {selected_local}: {exc}")

    if st.session_state.dataset is not None and st.button(
        "🔄 Reset / Reload Original File", use_container_width=True
    ):
        st.session_state.current_file_id = None
        st.rerun()

    st.divider()
    st.header("2. Baseline & Peak Settings")
    apply_zero = st.checkbox(
        "Zero spectra at reference wavelength",
        value=False,
        help="Subtracts A(λ_ref) from each spectrum so A(λ_ref) = 0.",
    )
    zero_ref_wl = st.number_input(
        "Zero reference wavelength (nm)",
        min_value=200.0,
        max_value=900.0,
        value=490.0,
        step=1.0,
        disabled=not apply_zero,
    )

    peak_prominence = st.slider(
        "Peak / Trough Prominence",
        min_value=0.001,
        max_value=0.100,
        value=0.005,
        step=0.001,
        format="%.3f",
        help="Minimum vertical prominence required for scipy.signal.find_peaks to mark a peak or trough.",
    )
    col_pk, col_tr = st.columns(2)
    mark_peaks = col_pk.checkbox("Mark Peaks (▲)", value=True)
    mark_troughs = col_tr.checkbox("Mark Troughs (▼)", value=True)

ds_base: SpectraDataset | None = st.session_state.dataset
if ds_base is None:
    st.info("👈 Upload or select a UV-Vis CSV file in the sidebar to begin.")
    st.stop()

# Apply non-destructive baseline zeroing if toggled
active_ds = (
    ds_base.zero_spectra(reference_wl=zero_ref_wl, in_place=False)
    if apply_zero
    else ds_base.copy()
)

# Info bar
info_cols = st.columns(4)
info_cols[0].metric("Dataset", active_ds.source_name)
info_cols[1].metric(
    "CSV Format",
    "Full Report" if active_ds.format_type == "full_report" else "Table Export",
)
info_cols[2].metric(
    "Wavelength Range",
    f"{active_ds.df['wavelength'].min():.0f}–{active_ds.df['wavelength'].max():.0f} nm",
)
info_cols[3].metric(
    "Spectra Count",
    f"{len(active_ds.raw_sample_names)} raw, {len(active_ds.difference_columns)} diff",
)

tab_inspect, tab_diff, tab_pub = st.tabs(
    [
        "🔍 Step 1: Inspect & Rename Spectra",
        "➖ Step 2: Difference Spectra & Concentration",
        "📄 Step 3: Publication Plot Export",
    ]
)

# ==============================================================================
# TAB 1: INSPECT & RENAME SPECTRA
# ==============================================================================
with tab_inspect:
    col_sel, col_ren = st.columns([2, 1])
    with col_sel:
        selected_inspect_samples = st.multiselect(
            "Select spectra to display:",
            options=active_ds.sample_names,
            default=active_ds.sample_names,
        )

    with col_ren:
        with st.expander("✏️ Rename Sample Labels"):
            st.caption("Give generic samples descriptive names (e.g., Fe(III), Fe(II), CO):")
            rename_map: dict[str, str] = {}
            for s_name in ds_base.raw_sample_names:
                new_val = st.text_input(
                    f"Rename '{s_name}'", value=s_name, key=f"ren_{s_name}"
                )
                if new_val.strip() and new_val.strip() != s_name:
                    rename_map[s_name] = new_val.strip()
            if st.button("Apply New Names", type="primary"):
                ds_base.rename_samples(rename_map, in_place=True)
                st.rerun()

    fig_interactive = plot_interactive(
        active_ds,
        samples=selected_inspect_samples,
        difference_samples=None,
        title=f"Interactive Spectra — {active_ds.source_name}",
        mark_peaks=mark_peaks,
        mark_troughs=mark_troughs,
        peak_prominence=peak_prominence,
    )
    st.plotly_chart(fig_interactive, use_container_width=True)

    html_buf = fig_interactive.to_html(include_plotlyjs="cdn")
    st.download_button(
        "⬇️ Download Interactive Plot (.html)",
        data=html_buf,
        file_name=f"{active_ds.source_name}_interactive.html",
        mime="text/html",
    )

    col_ext, col_df = st.columns(2)
    with col_ext:
        st.subheader("Detected Peaks (▲) & Troughs (▼)")
        extrema_df = active_ds.detect_extrema(
            samples=selected_inspect_samples,
            prominence=peak_prominence,
            find_positive_peaks=mark_peaks,
            find_negative_troughs=mark_troughs,
        )
        st.dataframe(extrema_df, use_container_width=True)

    with col_df:
        st.subheader("Polars DataFrame Preview")
        st.dataframe(active_ds.df, use_container_width=True)

# ==============================================================================
# TAB 2: DIFFERENCE SPECTRA & PROTEIN CONCENTRATION
# ==============================================================================
with tab_diff:
    st.subheader("Compute & Store Difference Spectrum (Sample A − Sample B)")
    d_col1, d_col2, d_col3, d_col4 = st.columns([1.2, 1.2, 1.4, 1.0])

    all_cols = active_ds.sample_names
    default_a = min(2, len(all_cols) - 1) if len(all_cols) > 1 else 0
    default_b = min(1, len(all_cols) - 1) if len(all_cols) > 1 else 0

    minuend = d_col1.selectbox(
        "Minuend (Sample A)", options=all_cols, index=default_a
    )
    subtrahend = d_col2.selectbox(
        "Subtrahend (Sample B)", options=all_cols, index=default_b
    )
    custom_diff_name = d_col3.text_input(
        "Difference Spectrum Label",
        value=f"{minuend} - {subtrahend}",
    )
    zero_diff_opt = d_col4.checkbox("Zero diff at 490 nm", value=False)

    if st.button("➕ Compute & Store Difference Spectrum", type="primary"):
        diff_col_created = ds_base.compute_difference(
            minuend=minuend,
            subtrahend=subtrahend,
            name=custom_diff_name,
            zero_at=490.0 if zero_diff_opt else None,
        )
        st.success(f"Stored difference spectrum: **{diff_col_created}**")
        st.rerun()

    # Refresh active_ds after any stored differences
    active_ds = (
        ds_base.zero_spectra(reference_wl=zero_ref_wl, in_place=False)
        if apply_zero
        else ds_base.copy()
    )

    if active_ds.difference_columns:
        st.markdown("#### Interactive Difference Plot")
        show_raw_with_diff = st.checkbox(
            "Show two-panel view (Raw spectra on top, Difference spectra on bottom)",
            value=True,
        )
        raw_for_diff_plot = (
            st.multiselect(
                "Raw spectra to show in top panel:",
                options=active_ds.raw_sample_names,
                default=[
                    s
                    for s in (minuend, subtrahend)
                    if s in active_ds.raw_sample_names
                ]
                or active_ds.raw_sample_names[:2],
            )
            if show_raw_with_diff
            else None
        )
        selected_diffs = st.multiselect(
            "Difference spectra to plot:",
            options=active_ds.difference_columns,
            default=active_ds.difference_columns,
        )

        fig_diff_int = plot_interactive(
            active_ds,
            samples=raw_for_diff_plot if show_raw_with_diff else None,
            difference_samples=selected_diffs,
            title=f"Difference Spectra — {active_ds.source_name}",
            mark_peaks=mark_peaks,
            mark_troughs=mark_troughs,
            peak_prominence=peak_prominence,
        )
        st.plotly_chart(fig_diff_int, use_container_width=True)

        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "⬇️ Download Difference Interactive Plot (.html)",
            data=fig_diff_int.to_html(include_plotlyjs="cdn"),
            file_name=f"{active_ds.source_name}_difference.html",
            mime="text/html",
        )
        csv_bytes = active_ds.df.write_csv().encode("utf-8")
        dl_col2.download_button(
            "⬇️ Download Polars Dataset with Differences (.csv)",
            data=csv_bytes,
            file_name=f"{active_ds.source_name}_with_differences.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("🧪 Protein Concentration Calculator")
    st.caption(
        r"Calculates $C = \text{dilution} \times \frac{A(\lambda_{\text{target}}) - A(\lambda_{\text{ref}})}{\varepsilon \cdot l}$ "
        r"(or single-wavelength $\frac{A(\lambda_{\text{target}})}{\varepsilon \cdot l}$ if reference subtraction is unchecked). "
        "Works on either a difference spectrum or a raw sample spectrum."
    )

    c_col1, c_col2, c_col3 = st.columns(3)

    # Default to the most recently added difference spectrum if available
    conc_options = (
        list(reversed(active_ds.difference_columns)) + active_ds.raw_sample_names
    )
    conc_spectrum = c_col1.selectbox(
        "Spectrum to analyze:", options=conc_options, index=0
    )

    eps_preset = c_col1.selectbox(
        "Extinction Coefficient (ε) Preset:",
        options=[
            "P450 CO-Difference (91,000 M⁻¹cm⁻¹)",
            "Heme / P420 (44,500 M⁻¹cm⁻¹)",
            "Custom",
        ],
    )
    default_eps = (
        91000.0
        if "91,000" in eps_preset
        else (44500.0 if "44,500" in eps_preset else 91000.0)
    )
    ext_coeff_val = c_col1.number_input(
        "Extinction Coefficient (ε)",
        min_value=1.0,
        value=default_eps,
        step=500.0,
    )
    ext_coeff_unit = c_col1.selectbox(
        "ε Units:", options=["M^-1 cm^-1", "mM^-1 cm^-1"], index=0
    )

    target_mode = c_col2.radio(
        "Target Wavelength Selection:",
        options=["Peak in Wavelength Window", "Exact Wavelength"],
        index=0,
    )
    if target_mode == "Peak in Wavelength Window":
        w_col_a, w_col_b = c_col2.columns(2)
        win_min = w_col_a.number_input("Window Min (nm)", value=445.0, step=1.0)
        win_max = w_col_b.number_input("Window Max (nm)", value=455.0, step=1.0)
        target_wl_arg: float | tuple[float, float] = (win_min, win_max)
    else:
        exact_wl = c_col2.number_input(
            "Target Wavelength (nm)", value=450.0, step=1.0
        )
        target_wl_arg = exact_wl

    use_ref_wl = c_col2.checkbox(
        "Subtract baseline reference wavelength (A_ref)", value=True
    )
    ref_wl_val = c_col2.number_input(
        "Reference Wavelength (nm)",
        value=490.0,
        step=1.0,
        disabled=not use_ref_wl,
    )

    dilution_val = c_col3.number_input(
        "Dilution Factor (e.g. 50 for 1:50)", min_value=0.001, value=50.0, step=1.0
    )
    pathlength_val = c_col3.number_input(
        "Pathlength (cm)", min_value=0.01, value=1.0, step=0.1
    )
    out_unit = c_col3.selectbox("Output Concentration Unit:", options=["uM", "mM", "M"], index=0)

    try:
        conc_res = active_ds.calculate_concentration(
            spectrum=conc_spectrum,
            extinction_coeff=ext_coeff_val,
            target_wl=target_wl_arg,
            reference_wl=ref_wl_val if use_ref_wl else None,
            dilution_factor=dilution_val,
            pathlength_cm=pathlength_val,
            extinction_coeff_unit=ext_coeff_unit,  # type: ignore[arg-type]
            unit=out_unit,  # type: ignore[arg-type]
        )

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        unit_disp = "µM" if conc_res.unit == "uM" else conc_res.unit
        res_col1.metric(
            "Calculated Concentration",
            f"{conc_res.concentration:.2f} {unit_disp}",
        )
        res_col2.metric(
            f"A({conc_res.target_wavelength_nm:.0f} nm)",
            f"{conc_res.target_absorbance:.5f}",
        )
        if conc_res.reference_wavelength_nm is not None:
            res_col3.metric(
                f"A({conc_res.reference_wavelength_nm:.0f} nm)",
                f"{conc_res.reference_absorbance:.5f}",
            )
        else:
            res_col3.metric("Reference Subtracted", "None")
        res_col4.metric("Δ Absorbance", f"{conc_res.delta_absorbance:.5f}")

        st.code(conc_res.summary(), language="text")
    except Exception as exc:
        st.warning(f"Could not compute concentration: {exc}")

# ==============================================================================
# TAB 3: PUBLICATION PLOT EXPORT
# ==============================================================================
with tab_pub:
    st.subheader("Minimal Scientific Publication Plot (No Grid, Despined)")

    p_col1, p_col2 = st.columns([1.1, 1.9])
    with p_col1:
        pub_layout = st.radio(
            "Panel Layout:",
            options=[
                "Two-Panel (Raw + Difference)"
                if active_ds.difference_columns
                else "Single-Panel (Selected Spectra)",
                "Single-Panel (Selected Spectra)",
                "Single-Panel (Difference Only)",
            ],
            index=0,
        )

        if pub_layout == "Two-Panel (Raw + Difference)":
            pub_raw_samples = st.multiselect(
                "Top Panel (Raw Spectra):",
                options=active_ds.raw_sample_names,
                default=active_ds.raw_sample_names[:3],
                key="pub_raw_two",
            )
            pub_diff_samples = st.multiselect(
                "Bottom Panel (Difference Spectra):",
                options=active_ds.difference_columns,
                default=active_ds.difference_columns,
                key="pub_diff_two",
            )
        elif pub_layout == "Single-Panel (Difference Only)":
            pub_raw_samples = None
            pub_diff_samples = st.multiselect(
                "Difference Spectra to Plot:",
                options=active_ds.difference_columns or active_ds.sample_names,
                default=active_ds.difference_columns or active_ds.sample_names[:1],
                key="pub_diff_single",
            )
        else:
            pub_raw_samples = st.multiselect(
                "Spectra to Plot:",
                options=active_ds.sample_names,
                default=active_ds.sample_names[:4],
                key="pub_raw_single",
            )
            pub_diff_samples = None

        pub_title = st.text_input("Plot Title (leave blank for journal figures):", value="")
        pub_xlabel = st.text_input("X-Axis Label:", value="Wavelength (nm)")
        pub_ylabel = st.text_input("Y-Axis Label:", value="Absorbance")
        pub_lw = st.slider("Line Width:", min_value=0.8, max_value=3.5, value=1.8, step=0.1)
        f_col_w, f_col_h = st.columns(2)
        fig_w = f_col_w.number_input("Width (in)", min_value=3.0, max_value=14.0, value=6.5, step=0.5)
        fig_h = f_col_h.number_input(
            "Height (in)",
            min_value=2.5,
            max_value=14.0,
            value=6.8 if pub_layout == "Two-Panel (Raw + Difference)" else 4.2,
            step=0.5,
        )
        pm_col1, pm_col2 = st.columns(2)
        pub_mark_pk = pm_col1.checkbox("Annotate Peaks", value=False)
        pub_mark_tr = pm_col2.checkbox("Annotate Troughs", value=False)
        pub_zero_line = st.checkbox("Show y = 0 dotted line on difference panel", value=True)

    with p_col2:
        fig_pub = plot_publication(
            active_ds,
            samples=pub_raw_samples,
            difference_samples=pub_diff_samples,
            title=pub_title if pub_title.strip() else None,
            xlabel=pub_xlabel,
            ylabel=pub_ylabel,
            mark_peaks=pub_mark_pk,
            mark_troughs=pub_mark_tr,
            peak_prominence=peak_prominence,
            show_zero_line=pub_zero_line,
            linewidth=pub_lw,
            figsize=(fig_w, fig_h),
        )
        st.pyplot(fig_pub, use_container_width=False)

        # Prepare vector PDF, SVG, and 300-DPI PNG in memory for instant download
        buf_pdf = io.BytesIO()
        fig_pub.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        buf_svg = io.BytesIO()
        fig_pub.savefig(buf_svg, format="svg", bbox_inches="tight")
        buf_png = io.BytesIO()
        fig_pub.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig_pub)

        exp_c1, exp_c2, exp_c3 = st.columns(3)
        exp_c1.download_button(
            "⬇️ Download Vector PDF",
            data=buf_pdf.getvalue(),
            file_name=f"{active_ds.source_name}_publication.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        exp_c2.download_button(
            "⬇️ Download Vector SVG",
            data=buf_svg.getvalue(),
            file_name=f"{active_ds.source_name}_publication.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
        exp_c3.download_button(
            "⬇️ Download 300-DPI PNG",
            data=buf_png.getvalue(),
            file_name=f"{active_ds.source_name}_publication.png",
            mime="image/png",
            use_container_width=True,
        )
