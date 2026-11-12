"""Survival analysis and threshold-screening utilities.

General-purpose DataFrame APIs for patient-level threshold groups, KM estimates,
finite-horizon statistics, comparable plots, and LaTeX exports. No project-specific
models, cohorts, directories, or data loading are embedded here.

For reproducibility, risk_set_mode='original_filter' preserves an earlier workflow
that excludes observations with durations beyond the window before KM fitting.
This is not generally appropriate for survival inference; administrative censoring
is available as an explicit alternative and changes the KM estimates.
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
    "make_patient_level_threshold_dataset",
    "compute_km_statistics",
    "dynamic_threshold_metrics",
    "plot_km_comparison",
    "plot_finite_horizon_comparison",
    "survival_summary_to_latex",
    "export_survival_summary_table",
]


def _halve_x_margins(ax):
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
) -> tuple:
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


# -----------------------------------------------------------------------------
# Reusable multi-cohort plots. No project-specific paths or model IDs.
# -----------------------------------------------------------------------------

_DEFAULT_SURVIVAL_COLORS = {1: '#569D3E', 0: '#BE3D30'}
_DEFAULT_METRIC_COLORS = {
    'P_event_within_k_given_crossed': '#4675AF',
    'P_no_event_within_k_given_never_crossed': '#E88833',
    'P_crossed_given_event_within_k': '#569D3E',
    'P_never_crossed_given_no_event_within_k': '#BE3D30',
}
_DEFAULT_METRIC_LABELS = {
    'P_event_within_k_given_crossed': r'$P(event \leq k \mid crossed)$',
    'P_no_event_within_k_given_never_crossed': r'$P(no\ event \leq k \mid never\ crossed)$',
    'P_crossed_given_event_within_k': r'$P(crossed \mid event \leq k)$',
    'P_never_crossed_given_no_event_within_k': r'$P(never\ crossed \mid no\ event \leq k)$',
}


def _plot_order(frames, cohort_order):
    if not frames:
        raise ValueError('At least one cohort DataFrame is required')
    order = tuple(frames) if cohort_order is None else tuple(cohort_order)
    if not order:
        raise ValueError('cohort_order must not be empty')
    missing = [cohort for cohort in order if cohort not in frames]
    if missing:
        raise KeyError(f'No DataFrame supplied for cohort(s): {missing}')
    return order


def _save_show_close(fig, outfile, show, close, dpi):
    if outfile is not None:
        target = Path(outfile)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_km_comparison(
    patients_by_cohort,
    *,
    max_week=24,
    cohort_order=None,
    group_col='crossed_threshold',
    duration_col='duration',
    event_col='event_observed',
    group_order=(1, 0),
    group_labels=None,
    group_colors=None,
    cohort_linestyles=None,
    risk_set_mode='original_filter',
    ci_show=True,
    ci_alpha=0.18,
    linewidth=3.25,
    legend_fontsize=14,
    figsize=(8.2, 5.8),
    dpi=600,
    outfile=None,
    show=False,
    close=False,
):
    """Plot comparable KM curves across arbitrary named cohorts and groups.

    Each cohort is a DataFrame containing duration, event indicator, group.
    ``risk_set_mode='original_filter'`` reproduces the source notebook:
    retain only subjects with observed duration <= max_week before fitting.
    This can bias survival estimates. ``administrative`` retains subjects
    followed past the horizon and administratively censors them at max_week.

    Returns ``(figure, axes)``. Fits and confidence bands are unchanged
    relative to lifelines' KaplanMeierFitter for a given filtered dataset.
    """
    order = _plot_order(patients_by_cohort, cohort_order)
    colors = {**_DEFAULT_SURVIVAL_COLORS, **(group_colors or {})}
    labels = {1: 'crossed', 0: 'never crossed'}
    if group_labels:
        labels.update(group_labels)
    styles = {cohort: '-' for cohort in order}
    if cohort_linestyles:
        styles.update(cohort_linestyles)
    if risk_set_mode not in ('original_filter', 'administrative'):
        raise ValueError("risk_set_mode must be 'original_filter' or 'administrative'")
    if max_week <= 0:
        raise ValueError('max_week must be positive')

    fig, ax = plt.subplots(figsize=figsize)
    for group in group_order:
        if group not in colors:
            raise ValueError(f'Color unspecified for group {group!r}')
        for cohort in order:
            frame = patients_by_cohort[cohort]
            needed = [group_col, duration_col, event_col]
            absent = [name for name in needed if name not in frame]
            if absent:
                raise ValueError(f'{cohort} missing columns {absent}')
            subset = frame.loc[frame[group_col] == group, needed].dropna().copy()
            total = len(subset)
            if not total:
                raise ValueError(f'{cohort} group {group} is empty')
            if risk_set_mode == 'original_filter':
                fit_frame = subset.loc[subset[duration_col] <= max_week]
                durations = fit_frame[duration_col]
                events = fit_frame[event_col]
            else:
                durations = subset[duration_col].clip(upper=max_week)
                events = (subset[event_col].astype(bool) &
                          (subset[duration_col] <= max_week)).astype(int)
            if len(durations) == 0:
                raise ValueError(f'{cohort} group {group} has no eligible durations')
            fit = KaplanMeierFitter()
            fit.fit(durations, event_observed=events,
                    label=f'{cohort}: {labels.get(group, group)} (n={total:,})')
            fit.plot_survival_function(ax=ax, color=colors[group],
                                       linestyle=styles[cohort], linewidth=linewidth,
                                       ci_show=ci_show, ci_alpha=ci_alpha)

    ax.set_xlabel('Weeks since screening', fontsize=18)
    ax.set_ylabel('Event-free survival probability', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, color='0.68', alpha=0.42, linewidth=0.95)
    ax.legend(frameon=False, loc='lower left', fontsize=legend_fontsize)
    _halve_x_margins(ax)
    fig.tight_layout()
    _save_show_close(fig, outfile, show, close, dpi)
    return fig, ax


def plot_finite_horizon_comparison(
    metrics_by_cohort,
    *,
    max_week=24,
    cohort_order=None,
    cohort_linestyles=None,
    landmark_weeks=None,
    metric_colors=None,
    metric_labels=None,
    metrics=None,
    week_col='k_weeks',
    linewidth=3.25,
    marker_size=6,
    legend_fontsize=12,
    figsize=(8.2, 5.8),
    dpi=600,
    outfile=None,
    show=False,
    close=False,
):
    """Plot finite-horizon metrics in multiple cohorts on a shared axis.

    The metrics are supplied as precomputed DataFrames; this routine neither
    reconstructs labels nor changes dynamic-metric estimation.
    """
    order = _plot_order(metrics_by_cohort, cohort_order)
    styles = {cohort: '-' for cohort in order}
    if cohort_linestyles:
        styles.update(cohort_linestyles)
    colors = {**_DEFAULT_METRIC_COLORS, **(metric_colors or {})}
    labels = {**_DEFAULT_METRIC_LABELS, **(metric_labels or {})}
    selected = tuple(_DEFAULT_METRIC_COLORS) if metrics is None else tuple(metrics)
    if not selected:
        raise ValueError('At least one dynamic metric must be selected')
    fig, ax = plt.subplots(figsize=figsize)

    for cohort in order:
        frame = metrics_by_cohort[cohort]
        required = [week_col, *selected]
        absent = [name for name in required if name not in frame]
        if absent:
            raise ValueError(f'{cohort} missing columns {absent}')
        use = frame.loc[frame[week_col].between(0, max_week)].sort_values(week_col)
        if landmark_weeks is not None:
            use = use.loc[use[week_col].isin(landmark_weeks)]
        for metric in selected:
            ax.plot(use[week_col], use[metric], color=colors[metric],
                    linestyle=styles[cohort], marker='o', markersize=marker_size,
                    linewidth=linewidth)

    handles = [Line2D([0], [0], color='black', linestyle=styles[cohort],
                      linewidth=linewidth, label=cohort) for cohort in order]
    handles += [Line2D([0], [0], color=colors[metric], linestyle='-',
                       linewidth=linewidth, label=labels.get(metric, metric))
                for metric in selected]
    ax.set_xlabel('Follow-up window k (weeks)', fontsize=18)
    ax.set_ylabel('Probability', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, color='0.68', alpha=0.42, linewidth=0.95)
    ax.legend(handles=handles, frameon=False, loc='lower right',
              bbox_to_anchor=(1, 0.465), fontsize=legend_fontsize)
    _halve_x_margins(ax)
    fig.tight_layout()
    _save_show_close(fig, outfile, show, close, dpi)
    return fig, ax


def _parse_specificity(value):
    """Accept 0.975, '0.975', or historical '0975' encoding."""
    if isinstance(value, str) and value.startswith('0') and '.' not in value and len(value) > 1:
        result = float('0.' + value[1:])
    else:
        result = float(value)
    if not 0 < result <= 1:
        raise ValueError('specificity must be in (0,1]')
    return result


def survival_summary_to_latex(
    summary_df,
    *,
    horizon,
    specificity,
    targets,
    cohorts,
    target_col='M',
    cohort_col='source',
    label='table_SURVIVAL',
    caption=None,
):
    """Format the supplied threshold-enrichment summary as a LaTeX table.

    Expects summary_df to contain precomputed quantities. The library does not
    read project-specific CSVs or select hard-coded targets/cohorts.
    """
    target_order = tuple(targets)
    source_order = tuple(cohorts)
    if not target_order or not source_order:
        raise ValueError('targets and cohorts must be nonempty')
    spec_pct = f'{100 * _parse_specificity(specificity):g}'
    table_df = summary_df.set_index([target_col, cohort_col], verify_integrity=True)
    if caption is None:
        source_names = ' and '.join(source_order)
        caption = (f'Threshold-based event enrichment in {source_names} at '
                   f'{horizon} weeks using the {spec_pct}\\%-specificity operating point')
    metrics = (
        ('threshold', r'\shortstack{Risk\\threshold}', 'threshold'),
        ('n_crossed', r'\shortstack{Crossed\\/ total}', 'count'),
        ('crossing_rate_pct', r'\shortstack{Crossing\\rate}', 'percent'),
        ('event_probability_crossed_pct', r'\shortstack{Event probability\\if crossed}', 'percent'),
        ('event_probability_never_crossed_pct', r'\shortstack{Event probability\\if never crossed}', 'percent'),
        ('risk_ratio', 'RR', 'number'),
        ('ARD_pp', r'\shortstack{ARD\\(pp)}', 'number'),
        ('logrank_p', r'\shortstack{Log-rank\\\textit{p}}', 'pvalue'),
    )
    headers = ['Event', 'Source', *(item[1] for item in metrics)]
    lines = [
        r'\begin{table}[ht!]', r'\centering', rf'\caption{{{caption}}}',
        rf'\label{{{label}}}', r'\normalfont\fontsize{11}{13}\selectfont', '',
        r'\resizebox{\textwidth}{!}{%',
        r'% Reduce horizontal padding to allow larger text within the page width.',
        r'\setlength{\tabcolsep}{3pt}%',
        r'% Local vertical padding for this table only.',
        r'\renewcommand{\arraystretch}{1.25}%',
        r'\setlength{\extrarowheight}{2pt}%',
        r'\begin{tabular}{L{0.95in}lrrrrrrrr}', r'\hline',
        ' &\n'.join(rf'\raisebox{{-0.5\height}}{{\textbf{{{h}}}}}' for h in headers) + r' \\[3pt]',
        r'\hline',
    ]
    for target in target_order:
        event_name = str(target).replace('_', ' ')
        lines.append(rf'\multirow{{{len(source_order)}}}{{0.95in}}{{\raggedright {event_name}}}')
        for cohort in source_order:
            row = table_df.loc[(target, cohort)]
            vals = []
            for column, _, style in metrics:
                value = row[column]
                if pd.isna(value):
                    text = '--'
                elif style == 'count':
                    text = f"{int(value):,} / {int(row['n_total']):,}"
                elif style == 'threshold':
                    text = f'{value:.4f}'
                elif style == 'percent':
                    text = rf'{value:.1f}\%'
                elif style == 'number':
                    text = f'{value:.2f}'
                elif value == 0:
                    text = r'$<$10\textsuperscript{-300}'
                else:
                    mantissa, exponent = f'{value:.2e}'.split('e')
                    text = rf'{mantissa}\,$\times$\,10\textsuperscript{{{int(exponent)}}}'
                vals.append(text)
            lines.append(' & ' + ' & '.join([cohort, *vals]) + r' \\')
        lines.append(r'\hline')

    zero_p = pd.to_numeric(table_df['logrank_p'], errors='coerce').eq(0)
    zero_sources = table_df.loc[zero_p].index.get_level_values(cohort_col).unique()
    p_note = r'\textit{p}: log-rank test p-value.'
    if len(zero_sources):
        p_note += (f" {' and '.join(zero_sources)} values that underflowed to zero in the saved output "
                   r'are reported as $p<10^{-300}$.')
    lines.extend([
        r'\end{tabular}%', r'}', '', r'\vskip 1em',
        r'\normalfont\fontsize{10}{12}\selectfont', r'\raggedright',
        f'Event probabilities are observed proportions within {horizon} weeks after screening.',
        rf'Both datasets use their saved operating threshold corresponding to {spec_pct}\% specificity in retrospective validation.',
        'RR: risk ratio between crossed and never-crossed groups within each source.',
        'ARD: absolute risk difference; pp: percentage points.',
        p_note, r'\par', r'\end{table}',
    ])
    return '\n'.join(lines) + '\n'


def export_survival_summary_table(summary_df, outfile, **kwargs):
    """Write a general precomputed summary table and return the output path."""
    text = survival_summary_to_latex(summary_df, **kwargs)
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path
