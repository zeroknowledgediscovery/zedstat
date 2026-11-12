"""Survival screening analysis and paired-cohort plotting helpers.

This module preserves the estimators and figure construction used in the
original KM_PLOTS.ipynb (2026-11-12). ``plot_joined_km`` intentionally retains
the original ``duration <= max_week`` PRE-FILTER. That is not equivalent to
administrative censoring at ``max_week``. It is retained for numerical and
figure parity, not because it is recommended for a prospective survival study.

Note: crossed/never-crossed index times are not necessarily aligned under
``make_patient_level_threshold_dataset``. Users should validate their study
design before causal or prospective interpretation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

__all__ = [
    'make_patient_level_threshold_dataset',
    'compute_km_statistics',
    'dynamic_threshold_metrics',
    'plot_km_paper',
    'plot_dynamic_metrics_paper',
    'plot_joined_km',
    'plot_joined_dynamic_metrics',
    'export_joined_survival_summary_table',
]

def _apply_large_paper_fonts(
    fig,
    ax,
    label_fontsize=18,
    tick_fontsize=15,
    legend_fontsize=14,
    title_fontsize=18,
    at_risk_fontsize=13,
):
    """Apply larger font sizes to all visible text in a matplotlib/lifelines figure."""
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    ax.title.set_size(title_fontsize)

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize - 1)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)

    # Lifelines at-risk table is added as an additional axis;
    # this makes its text legible too.
    for axis in fig.axes:
        axis.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        axis.tick_params(axis="both", which="minor", labelsize=tick_fontsize - 1)

        for text in axis.texts:
            text.set_fontsize(at_risk_fontsize)

        axis.xaxis.label.set_size(label_fontsize)
        axis.yaxis.label.set_size(label_fontsize)
        axis.title.set_size(title_fontsize)


def plot_km_paper(
    patient_df,
    title="Event-free survival after threshold-based positive screen",
    subtitle=None,
    duration_col="duration",
    event_col="event_observed",
    group_col="crossed_threshold",
    max_week=24,
    show_ci=True,
    show_at_risk=True,
    outfile=None,
    figsize=(8.2, 5.8),
    label_fontsize=18,
    tick_fontsize=15,
    legend_fontsize=16,
    title_fontsize=18,
    at_risk_fontsize=13,
    line_width=4,
    grid_alpha=0.42,
    grid_linewidth=0.95,
    grid_color="0.68",
):
    d = patient_df[[duration_col, event_col, group_col]].dropna().copy()
    d[event_col] = d[event_col].astype(int)
    d[group_col] = d[group_col].astype(int)

    crossed = d[d[group_col] == 1]
    never = d[d[group_col] == 0]

    if crossed.empty or never.empty:
        raise ValueError("Both crossed and never-crossed groups must be nonempty.")

    km_crossed = KaplanMeierFitter()
    km_never = KaplanMeierFitter()

    km_crossed.fit(
        crossed[duration_col],
        event_observed=crossed[event_col],
        label=f"Crossed threshold (n={len(crossed):,})",
    )

    km_never.fit(
        never[duration_col],
        event_observed=never[event_col],
        label=f"Never crossed threshold (n={len(never):,})",
    )

    fig, ax = plt.subplots(figsize=figsize)

    km_crossed.plot_survival_function(
        ax=ax,
        ci_show=show_ci,
        linewidth=line_width,
    )

    km_never.plot_survival_function(
        ax=ax,
        ci_show=show_ci,
        linewidth=line_width,
    )

    ax.set_xlim(0, max_week)
    ax.set_ylim(0, 1.0)

    ax.set_xlabel("Weeks since screening", fontsize=label_fontsize)
    ax.set_ylabel("Event-free survival probability", fontsize=label_fontsize)

    # Darker grid/tick guide lines on the plot face.
    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
        color=grid_color,
    )

    ax.legend(frameon=False, loc="lower left", fontsize=legend_fontsize)

    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize - 1)

    if show_at_risk:
        add_at_risk_counts(km_crossed, km_never, ax=ax)
        fig.subplots_adjust(bottom=0.28)
    else:
        fig.tight_layout()

    _apply_large_paper_fonts(
        fig=fig,
        ax=ax,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        title_fontsize=title_fontsize,
        at_risk_fontsize=at_risk_fontsize,
    )

    # Re-apply grid after lifelines/formatting calls.
    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
        color=grid_color,
    )

    if not show_at_risk:
        fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight")

    return fig, ax


def plot_dynamic_metrics_paper(
    metrics: pd.DataFrame,
    title: str,
    outfile: str | Path | None = None,
    figsize=(8.2, 5.8),
    label_fontsize=18,
    tick_fontsize=15,
    legend_fontsize=16,
    title_fontsize=18,
    line_width=3.8,
    marker_size=8,
    spine_linewidth=1.1,
    grid_alpha=0.42,
    grid_linewidth=0.95,
    grid_color="0.68",
):
    """Plot dynamic threshold metrics as paper-quality line plot with large fonts."""
    fig, ax = plt.subplots(figsize=figsize)
    x = metrics["k_weeks"]

    plot_cols = [
        ("P_event_within_k_given_crossed", r"$P(event \leq k \mid crossed)$"),
        ("P_no_event_within_k_given_never_crossed", r"$P(no\ event \leq k \mid never\ crossed)$"),
        ("P_crossed_given_event_within_k", r"$P(crossed \mid event \leq k)$"),
        ("P_never_crossed_given_no_event_within_k", r"$P(never\ crossed \mid no\ event \leq k)$"),
    ]

    for col, label in plot_cols:
        ax.plot(
            x,
            metrics[col],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            label=label,
        )

    ax.set_xlabel("Follow-up window k (weeks)", fontsize=label_fontsize)
    ax.set_ylabel("Probability", fontsize=label_fontsize)
    ax.set_ylim(0, 1.0)

    print(f"TITLE :: {title}")
    # ax.set_title(title, fontsize=title_fontsize, pad=10)

    # Keep the full plot box/frame visible.
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(spine_linewidth)

    # Darker grid/tick guide lines on the plot face.
    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
        color=grid_color,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        top=False,
        right=False,
        direction="out",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=tick_fontsize - 1,
        top=False,
        right=False,
        direction="out",
    )

    ax.legend(
        frameon=False,
        fontsize=legend_fontsize,
        loc="best",
        handlelength=2.2,
        borderaxespad=0.4,
        labelspacing=0.45,
    )

    _apply_large_paper_fonts(
        fig=fig,
        ax=ax,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        title_fontsize=title_fontsize,
    )

    # Re-apply visible spines and grid after helper, in case any style/helper changes them.
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(spine_linewidth)

    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
        color=grid_color,
    )

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight")

    return fig, ax


def halve_x_margins(ax):
    """ Make the plot more compact by reducing whitespace around the lines in the plot's face """
    data_left, data_right = ax.dataLim.intervalx
    view_left, view_right = ax.get_xlim()

    ax.set_xlim(
        data_left - (data_left - view_left) / 2,
        data_right + (view_right - data_right) / 2,
    )

def make_patient_level_threshold_dataset(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "risk",
    patient_col: str = "patient_id",
    duration_col: str = "week",
    time_col: str = "time",
    event_observed_col: str = "event_observed",
    baseline_week: int = -26,
    crossing_rule: str = "first",
    zero_duration_value: float = 1e-3,
    analysis_mode: str = "full",
) -> pd.DataFrame:
    """
    Collapse weekly rows to one row per patient.

    Parameters
    ----------
    analysis_mode : {"full", "event_only"}
        "full": retain event and non-event patients. This is the deployment-valid analysis.
        "event_only": retain only patients who eventually experience the event. This is a case-only
        timing analysis and should not be interpreted as PPV/NPV/specificity.

    crossing_rule : {"first", "last", "max"}
        Which threshold-crossing row to use as the patient index time.
    """
    if patient_col not in df.columns:
        d = df.reset_index()
    else:
        d = df.copy()

    required = {patient_col, duration_col, score_col, time_col, event_observed_col}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = d[[patient_col, duration_col, score_col, time_col, event_observed_col]].copy()
    d = d.dropna(subset=[patient_col, duration_col, score_col, time_col])
    d = d.sort_values([patient_col, duration_col])

    # Patient-level event status, used for optional event-only filtering.
    event_status = d.groupby(patient_col)[event_observed_col].max().astype(int)
    if analysis_mode == "event_only":
        keep_ids = set(event_status[event_status > 0].index)
        d = d[d[patient_col].isin(keep_ids)].copy()
    elif analysis_mode != "full":
        raise ValueError('analysis_mode must be "full" or "event_only"')

    rows = []
    for pid, g in d.groupby(patient_col, sort=False):
        g = g.sort_values(duration_col).copy()
        event_observed = int(g[event_observed_col].max() > 0)
        crossing = g[g[score_col] > threshold]

        if not crossing.empty:
            crossed_threshold = 1
            if crossing_rule == "first":
                idx_row = crossing.iloc[0]
            elif crossing_rule == "last":
                idx_row = crossing.iloc[-1]
            elif crossing_rule == "max":
                idx_row = crossing.loc[crossing[score_col].idxmax()]
            else:
                raise ValueError('crossing_rule must be "first", "last", or "max"')
        else:
            crossed_threshold = 0
            baseline_rows = g[g[duration_col] == baseline_week]
            idx_row = baseline_rows.iloc[0] if not baseline_rows.empty else g.iloc[0]

        raw_duration = -float(idx_row[duration_col])
        km_duration = max(raw_duration, zero_duration_value)
        discrete_duration = max(raw_duration, 1.0)


        rows.append({
            patient_col: pid,
            "crossed_threshold": crossed_threshold,
            "never_crossed_threshold": 1 - crossed_threshold,
            "risk_group": "Crossed threshold" if crossed_threshold else "Never crossed threshold",
            "index_week": float(idx_row[duration_col]),
            "index_score": float(idx_row[score_col]),
            "raw_duration": raw_duration,
            "duration": km_duration,
            "duration_discrete": discrete_duration,
            "event_observed": event_observed,
        })

    out = pd.DataFrame(rows)
    out["event_observed"] = out["event_observed"].astype(int)
    out["crossed_threshold"] = out["crossed_threshold"].astype(int)
    return out


def compute_km_statistics(
    patient_df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event_observed",
    group_col: str = "crossed_threshold",
    landmark_weeks=(4, 8, 10),
) -> tuple[dict, pd.DataFrame]:
    """
    Compute KM/log-rank statistics without fitting Cox PH.

    This avoids Cox convergence failures when the threshold creates near-complete
    separation or highly discretized event times.

    Returns
    -------
    stats : dict
        Overall counts and log-rank p-value.

    landmarks : pd.DataFrame
        Cumulative incidence, risk ratio, risk difference, and dynamic
        sensitivity/specificity-like quantities at each landmark week.
    """

    d = patient_df[[duration_col, event_col, group_col]].dropna().copy()
    d[event_col] = d[event_col].astype(int)
    d[group_col] = d[group_col].astype(int)

    high = d[d[group_col] == 1]
    low = d[d[group_col] == 0]

    if high.empty or low.empty:
        raise ValueError("Both threshold-crossing and never-crossing groups must be nonempty.")

    # Log-rank test
    lr = logrank_test(
        high[duration_col],
        low[duration_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col],
    )

    # KM fit
    km_high = KaplanMeierFitter()
    km_low = KaplanMeierFitter()

    km_high.fit(
        high[duration_col],
        event_observed=high[event_col],
        label="Crossed threshold",
    )

    km_low.fit(
        low[duration_col],
        event_observed=low[event_col],
        label="Never crossed threshold",
    )

    landmark_rows = []

    for k in landmark_weeks:
        s_high = float(km_high.predict(k))
        s_low = float(km_low.predict(k))

        ci_high = 1.0 - s_high
        ci_low = 1.0 - s_low

        # Direct discrete event status by k weeks
        event_within_k = (
            (d[event_col] == 1)
            & (d[duration_col] <= k)
        )

        no_event_within_k = ~event_within_k

        crossed = d[group_col] == 1
        never_crossed = d[group_col] == 0

        tp = int((crossed & event_within_k).sum())
        fp = int((crossed & no_event_within_k).sum())
        fn = int((never_crossed & event_within_k).sum())
        tn = int((never_crossed & no_event_within_k).sum())

        n_crossed = int(crossed.sum())
        n_never = int(never_crossed.sum())
        n_event_k = int(event_within_k.sum())
        n_no_event_k = int(no_event_within_k.sum())

        # Predictive-value-like quantities
        ppv_k = tp / n_crossed if n_crossed > 0 else np.nan
        npv_k = tn / n_never if n_never > 0 else np.nan

        # Sensitivity/specificity-like quantities
        sens_k = tp / n_event_k if n_event_k > 0 else np.nan
        spec_k = tn / n_no_event_k if n_no_event_k > 0 else np.nan

        # Fixed-window risk ratio and absolute risk difference
        risk_high_direct = tp / n_crossed if n_crossed > 0 else np.nan
        risk_low_direct = fn / n_never if n_never > 0 else np.nan

        risk_ratio = (
            risk_high_direct / risk_low_direct
            if risk_low_direct and risk_low_direct > 0
            else np.inf
        )
        risk_difference = risk_high_direct - risk_low_direct

        # Dynamic likelihood ratios
        fpr_k = 1.0 - spec_k if not np.isnan(spec_k) else np.nan
        fnr_k = 1.0 - sens_k if not np.isnan(sens_k) else np.nan

        lr_plus = sens_k / fpr_k if fpr_k and fpr_k > 0 else np.inf
        lr_minus = fnr_k / spec_k if spec_k and spec_k > 0 else np.nan

        landmark_rows.append({
            "week": k,

            "km_survival_crossed": s_high,
            "km_survival_never_crossed": s_low,
            "km_cumulative_incidence_crossed": ci_high,
            "km_cumulative_incidence_never_crossed": ci_low,

            "TP_k": tp,
            "FP_k": fp,
            "FN_k": fn,
            "TN_k": tn,

            "n_crossed": n_crossed,
            "n_never_crossed": n_never,
            "n_event_within_k": n_event_k,
            "n_no_event_within_k": n_no_event_k,

            "P_event_within_k_given_crossed": ppv_k,
            "P_no_event_within_k_given_never_crossed": npv_k,

            "P_crossed_given_event_within_k": sens_k,
            "P_never_crossed_given_no_event_within_k": spec_k,

            "risk_crossed": risk_high_direct,
            "risk_never_crossed": risk_low_direct,
            "risk_ratio": risk_ratio,
            "risk_difference": risk_difference,

            "dynamic_LR_plus": lr_plus,
            "dynamic_LR_minus": lr_minus,
        })

    stats = {
        "n_total": int(len(d)),
        "n_crossed": int(len(high)),
        "n_never_crossed": int(len(low)),
        "events_crossed": int(high[event_col].sum()),
        "events_never_crossed": int(low[event_col].sum()),
        "logrank_statistic": float(lr.test_statistic),
        "logrank_p": float(lr.p_value),
    }

    landmarks = pd.DataFrame(landmark_rows)

    return stats, landmarks


def dynamic_threshold_metrics(
    patient_df: pd.DataFrame,
    k_weeks=(1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26),
    duration_col: str = "duration_discrete",
    event_col: str = "event_observed",
    crossed_col: str = "crossed_threshold",
) -> pd.DataFrame:
    """Compute dynamic predictive values and operating characteristics by follow-up window."""
    d = patient_df[[duration_col, event_col, crossed_col]].copy()
    d[event_col] = d[event_col].astype(int)
    d[crossed_col] = d[crossed_col].astype(int)

    rows = []
    for k in k_weeks:
        crossed = d[crossed_col] == 1
        never = d[crossed_col] == 0
        event_k = (d[event_col] == 1) & (d[duration_col] <= k)
        no_event_k = ~event_k

        tp = int((crossed & event_k).sum())
        fp = int((crossed & no_event_k).sum())
        fn = int((never & event_k).sum())
        tn = int((never & no_event_k).sum())

        n_crossed = int(crossed.sum())
        n_never = int(never.sum())
        n_event_k = int(event_k.sum())
        n_no_event_k = int(no_event_k.sum())

        ppv_k = tp / n_crossed if n_crossed else np.nan
        npv_k = tn / n_never if n_never else np.nan
        se_k = tp / n_event_k if n_event_k else np.nan
        sp_k = tn / n_no_event_k if n_no_event_k else np.nan
        fpr_k = fp / n_no_event_k if n_no_event_k else np.nan
        fnr_k = fn / n_event_k if n_event_k else np.nan

        rows.append({
            "k_weeks": k,
            "n_total": int(len(d)),
            "n_crossed_threshold": n_crossed,
            "n_never_crossed_threshold": n_never,
            "n_event_within_k": n_event_k,
            "n_no_event_within_k": n_no_event_k,
            "TP_k": tp,   "FP_k": fp,
            "FN_k": fn,   "TN_k": tn,
            "P_event_within_k_given_crossed": ppv_k,
            "P_no_event_within_k_given_never_crossed": npv_k,
            "P_crossed_given_event_within_k": se_k,
            "P_never_crossed_given_no_event_within_k": sp_k,
            "dynamic_LR_plus": (se_k / fpr_k) if fpr_k and fpr_k > 0 else np.inf,
            "dynamic_LR_minus": (fnr_k / sp_k) if sp_k and sp_k > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_joined_km(
    patients_by_source,
    model,
    output_dir,
    *,
    max_week=24,
    colors=None,
    source_linestyles=None,
    linewidth=3.25,
    legend_fontsize=14,
    dpi=600,
    show=True,
):
    """Reproduce original MarketScan/UKHC joined KM plot with original selection.

    Inputs map cohort name to a dataframe with ``crossed_threshold``,
    ``duration``, and ``event_observed``. Returns the saved PNG path.

    IMPORTANT: original code excludes duration > max_week before fitting,
    retaining original published behavior; this is NOT administrative censoring.
    """
    PATIENTS = patients_by_source
    M = model
    SUBTITLE_WEEK = max_week
    OUTDIR_PAPER = str(output_dir)
    Path(OUTDIR_PAPER).mkdir(parents=True, exist_ok=True)
    COLORS = {'green': '#569D3E', 'red': '#BE3D30'} if colors is None else colors
    SOURCE_LINESTYLES = {'MarketScan': '-', 'UKHC': '--'} if source_linestyles is None else source_linestyles
    LINEWIDTH = linewidth
    LEGEND_FONTSIZE = legend_fontsize
    DPI = dpi
    fig, ax = plt.subplots(figsize = (8.2, 5.8))

    for SOURCE, CROSSED, COLOR, NAME in [
        ("UKHC", 1, COLORS["green"], "UKHC: crossed"),
        ("MarketScan", 1, COLORS["green"], "MarketScan: crossed"),
        ("UKHC", 0, COLORS["red"], "UKHC: never crossed"),
        ("MarketScan", 0, COLORS["red"], "MarketScan: never crossed"),
    ]:
        PREDS = PATIENTS[SOURCE]

        PREDS = PREDS[PREDS["crossed_threshold"] == CROSSED]
        PREDS_PLOT_SUB = PREDS[
            PREDS.duration <= SUBTITLE_WEEK
        ]

        KM = KaplanMeierFitter()
        KM.fit(
            PREDS_PLOT_SUB["duration"],
            event_observed = PREDS_PLOT_SUB["event_observed"],
            label = f"{NAME} (n={len(PREDS):,})",
        )
        KM.plot_survival_function(
            ax = ax,
            color = COLOR,
            linestyle = SOURCE_LINESTYLES[SOURCE],
            linewidth = LINEWIDTH,
            ci_show = True,
            ci_alpha = 0.18,
        )

    ax.set_xlabel("Weeks since screening", fontsize = 18)
    ax.set_ylabel("Event-free survival probability", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 15)
    ax.grid(True, color = "0.68", alpha = 0.42, linewidth = 0.95)
    ax.legend(frameon = False, loc = "lower left", fontsize = LEGEND_FONTSIZE)

    halve_x_margins(ax)
    fig.tight_layout()
    fig.savefig(
        f"{OUTDIR_PAPER}/km_{M}.png",
        dpi = DPI,
        bbox_inches = "tight",
        facecolor = "white",
    )
    if show:
        plt.show()
    plt.close(fig)
    return Path(OUTDIR_PAPER) / f"km_{M}.png"



def plot_joined_dynamic_metrics(
    metrics_by_source,
    model,
    output_dir,
    *,
    max_week=24,
    max_followup_weeks=26,
    landmark_weeks=(1,2,4,8,10,12,14,16,18,20,22,24,26),
    colors=None,
    source_linestyles=None,
    linewidth=3.25,
    marker_size=6,
    dpi=600,
    show=True,
):
    """Reproduce the original finite-horizon dynamic-metrics joined plot.

    The input metrics frames are the saved results produced by the original
    analysis; this function does not refit metrics or change censoring.
    """
    M = model
    OUTDIR_PAPER = str(output_dir)
    Path(OUTDIR_PAPER).mkdir(parents=True, exist_ok=True)
    SUBTITLE_WEEK = max_week
    MAX_FOLLOWUP_WEEKS = max_followup_weeks
    LANDMARK_WEEKS = landmark_weeks
    COLORS = {'blue':'#4675AF', 'orange':'#E88833', 'green':'#569D3E', 'red':'#BE3D30'} if colors is None else colors
    SOURCE_LINESTYLES = {'MarketScan': '-', 'UKHC': '--'} if source_linestyles is None else source_linestyles
    LINEWIDTH = linewidth
    MARKERSIZE = marker_size
    DPI = dpi
    METRIX = {
        SOURCE: DATA.loc[
            DATA['k_weeks'].isin(LANDMARK_WEEKS)
            & DATA['k_weeks'].between(0, MAX_FOLLOWUP_WEEKS)
        ].sort_values('k_weeks').copy()
        for SOURCE, DATA in metrics_by_source.items()
    }
    PLOT_COLS = [
        (
            "P_event_within_k_given_crossed",
            COLORS["blue"],
            r"$P(event \leq k \mid crossed)$",
        ),
        (
            "P_no_event_within_k_given_never_crossed",
            COLORS["orange"],
            r"$P(no\ event \leq k \mid never\ crossed)$",
        ),
        (
            "P_crossed_given_event_within_k",
            COLORS["green"],
            r"$P(crossed \mid event \leq k)$",
        ),
        (
            "P_never_crossed_given_no_event_within_k",
            COLORS["red"],
            r"$P(never\ crossed \mid no\ event \leq k)$",
        ),
    ]

    fig, ax = plt.subplots(figsize = (8.2, 5.8))

    for SOURCE in ["UKHC", "MarketScan"]:
        METRICS = METRIX[SOURCE]
        METRICS = METRICS[METRICS.k_weeks <= SUBTITLE_WEEK]

        for COL, COLOR, LABEL in PLOT_COLS:
            ax.plot(
                METRICS["k_weeks"],
                METRICS[COL],
                color = COLOR,
                linestyle = SOURCE_LINESTYLES[SOURCE],
                marker = "o",
                markersize = MARKERSIZE,
                linewidth = LINEWIDTH,
            )

    HANDLES = [
        Line2D(
            [0], [0],
            color = "black",
            linestyle = SOURCE_LINESTYLES[SOURCE],
            linewidth = LINEWIDTH,
            label = SOURCE,
        )
        for SOURCE in ["UKHC", "MarketScan"]
    ]
    HANDLES += [
        Line2D(
            [0], [0],
            color = COLOR,
            linestyle = "-",
            linewidth = LINEWIDTH,
            label = LABEL,
        )
        for COL, COLOR, LABEL in PLOT_COLS
    ]


    ax.set_xlabel("Follow-up window k (weeks)", fontsize = 18)
    ax.set_ylabel("Probability", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 15)
    ax.grid(True, color = "0.68", alpha = 0.42, linewidth = 0.95)
    ax.legend(
        handles = HANDLES,
        frameon = False,
        loc = "lower right",
        bbox_to_anchor = (1, 0.465),
        fontsize = 12,
    )

    halve_x_margins(ax)
    fig.tight_layout()
    fig.savefig(
        f"{OUTDIR_PAPER}/dynamic_metrics_{M}.png",
        dpi = DPI,
        bbox_inches = "tight",
        facecolor = "white",
    )
    if show:
        plt.show()
    plt.close(fig)
    return Path(OUTDIR_PAPER) / f"dynamic_metrics_{M}.png"



def export_joined_survival_summary_table(
    summary_df,
    output_dir,
    *,
    max_week=24,
    specificity='0975',
    print_latex=True,
):
    """Write the original manuscript's KM_SUMMARY_TABLE.tex without recalculation.

    Expects the original prefiltered summary dataframe (M, source, etc.).
    This is publication formatting, not a new statistical estimator.
    """
    SUMMARY_DF = summary_df
    OUTDIR_PAPER = str(output_dir)
    Path(OUTDIR_PAPER).mkdir(parents=True, exist_ok=True)
    SUBTITLE_WEEK = max_week
    STRICTER_SPEC = specificity
    WEEK = SUBTITLE_WEEK
    SPEC_CODE = str(STRICTER_SPEC).strip()

    SPECIFICITY_PCT = f"{100 * float('0.' + SPEC_CODE[1:]):g}"

    TARGET_ORDER = (
        "Pulmonary_Hypertension",
        "Oxygen_Dependence",
        "ARDS",
    )
    SOURCE_ORDER = ("MarketScan", "UKHC")

    METRICS = (
        ("threshold", r"\shortstack{Risk\\threshold}", "threshold"),
        ("n_crossed", r"\shortstack{Crossed\\/ total}", "count"),
        ("crossing_rate_pct", r"\shortstack{Crossing\\rate}", "percent"),
        (
            "event_probability_crossed_pct",
            r"\shortstack{Event probability\\if crossed}",
            "percent",
        ),
        (
            "event_probability_never_crossed_pct",
            r"\shortstack{Event probability\\if never crossed}",
            "percent",
        ),
        ("risk_ratio", "RR", "number"),
        ("ARD_pp", r"\shortstack{ARD\\(pp)}", "number"),
        ("logrank_p", r"\shortstack{Log-rank\\\textit{p}}", "pvalue"),
    )


    TABLE_DF = SUMMARY_DF.set_index(["M", "source"])

    HEADERS = ["Event", "Source", *(label for _, label, _ in METRICS)]

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        rf"\caption{{Threshold-based event enrichment in MarketScan and UKHC at {WEEK} weeks using the {SPECIFICITY_PCT}\%-specificity operating point}}",
        r"\label{table_SURVIVAL}",
        r"\normalfont\fontsize{11}{13}\selectfont",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"% Reduce horizontal padding to allow larger text within the page width.",
        r"\setlength{\tabcolsep}{3pt}%",
        r"% Local vertical padding for this table only.",
        r"\renewcommand{\arraystretch}{1.25}%",
        r"\setlength{\extrarowheight}{2pt}%",
        r"\begin{tabular}{L{0.95in}lrrrrrrrr}",
        r"\hline",
        " &\n".join(
            rf"\raisebox{{-0.5\height}}{{\textbf{{{label}}}}}"
            for label in HEADERS
        ) + r" \\[3pt]",
        r"\hline",
    ]

    for target in TARGET_ORDER:
        event_name = target.replace("_", " ")
        lines.append(rf"\multirow{{2}}{{0.95in}}{{\raggedright {event_name}}}")

        for source in SOURCE_ORDER:
            row = TABLE_DF.loc[(target, source)]
            values = []
            for column, _, style in METRICS:
                value = row[column]
                if pd.isna(value):
                    text = "--"
                elif style == "count":
                    text = f"{int(value):,} / {int(row['n_total']):,}"
                elif style == "threshold":
                    text = f"{value:.4f}"
                elif style == "percent":
                    text = rf"{value:.1f}\%"
                elif style == "number":
                    text = f"{value:.2f}"
                elif value == 0:
                    text = r"$<$10\textsuperscript{-300}"
                else:
                    mantissa, exponent = f"{value:.2e}".split("e")
                    text = rf"{mantissa}\,$\times$\,10\textsuperscript{{{int(exponent)}}}"
                values.append(text)
            lines.append(" & " + " & ".join([source, *values]) + r" \\")

        lines.append(r"\hline")

    zero_p = pd.to_numeric(TABLE_DF["logrank_p"], errors="coerce").eq(0)
    zero_sources = TABLE_DF.loc[zero_p].index.get_level_values("source").unique()

    p_note = r"\textit{p}: log-rank test p-value."
    if len(zero_sources):
        source_text = " and ".join(zero_sources)
        p_note += (
            rf" {source_text} values that underflowed to zero in the saved output "
            r"are reported as $p<10^{-300}$."
        )

    lines.extend([
        r"\end{tabular}%",
        r"}",
        "",
        r"\vskip 1em",
        r"\normalfont\fontsize{10}{12}\selectfont",
        r"\raggedright",
        f"Event probabilities are observed proportions within {WEEK} weeks after screening.",
        rf"Both datasets use their saved operating threshold corresponding to {SPECIFICITY_PCT}\% specificity in retrospective validation.",
        "RR: risk ratio between crossed and never-crossed groups within each source.",
        "ARD: absolute risk difference; pp: percentage points.",
        p_note,
        r"\par",
        r"\end{table}",
    ])

    LATEX = "\n".join(lines) + "\n"

    output_path = Path(OUTDIR_PAPER) / "KM_SUMMARY_TABLE.tex"
    output_path.write_text(LATEX, encoding="utf-8")

    if print_latex:
        print(LATEX, end="")
    return output_path
