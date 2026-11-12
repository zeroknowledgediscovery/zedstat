"""Reusable score-level curve construction, comparison plotting and operating metrics.

The numerical estimators are kept separate from cohort/file naming and manuscript
layouts. Inference uses the existing zedstat PRC and log-ratio CI implementations.
The LR frontier and fixed-specificity bootstrap methods retain their published
notebook algorithms; display smoothing is not a simultaneous confidence band.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, FuncFormatter
from sklearn.metrics import roc_curve

from . import zedstat as core
from .pr_ci import precision_recall_ci_from_scores, plot_precision_recall_ci_fill
from .ci_logratio import lr_logratio_ci

__all__ = [
    "lr_frontier_from_scores", "fixed_specificity_bootstrap", "score_curve_bundle",
    "plot_curve_panel", "performance_table_latex",
]


def lr_frontier_from_scores(data, *, score_col="predicted_risk", label_col="target",
                            min_count=10, fpr_min=0.005, fpr_max=0.95,
                            min_lrplus=1.001):
    """Same supported, nondominated LR operating points as the previous notebook.

    The previous notebook additionally ran 3,000 binomial bootstrap draws at
    every LR threshold, but discarded those CI results before making the final
    plot. They are omitted here; the plotted log-delta CIs are unchanged.
    """
    if not 0 < fpr_min < fpr_max < 1 or min_count < 1:
        raise ValueError("Invalid FPR range or minimum cell count")
    labels = pd.to_numeric(data[label_col], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(data[score_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(labels) & np.isfinite(scores) & np.isin(labels, (0, 1))
    labels, scores = labels[ok], scores[ok]
    positive_scores = np.sort(scores[labels == 1])
    negative_scores = np.sort(scores[labels == 0])
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if min(n_pos, n_neg) < 2 * min_count:
        raise ValueError(f"Too few positives or negatives: n_pos={n_pos}, n_neg={n_neg}")

    lo_fpr = max(float(fpr_min), 1 / n_neg)
    if lo_fpr >= fpr_max:
        raise ValueError("No FPR search range supports the specified sample size")
    knee = min(max(0.10, lo_fpr), fpr_max)
    levels = np.unique(np.r_[np.geomspace(lo_fpr, knee, 90),
                              np.linspace(knee, fpr_max, 170)])
    idx = np.clip(np.floor((1 - levels) * (n_neg - 1)).astype(int), 0, n_neg - 1)
    thresholds = np.unique(negative_scores[idx])[::-1]
    tp = n_pos - np.searchsorted(positive_scores, thresholds, side="left")
    fp = n_neg - np.searchsorted(negative_scores, thresholds, side="left")
    fn, tn = n_pos - tp, n_neg - fp
    tpr, fpr = tp / n_pos, fp / n_neg

    valid = ((tp >= min_count) & (fp >= min_count) &
             (fn >= min_count) & (tn >= min_count) &
             (fpr >= fpr_min) & (fpr <= fpr_max))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_plus = tpr / fpr
        lr_minus = (1 - tpr) / (1 - fpr)
    valid &= (np.isfinite(lr_plus) & np.isfinite(lr_minus) &
              (lr_plus > min_lrplus) & (lr_minus > 0) & (lr_minus < 1))
    out = pd.DataFrame({
        "LR-": lr_minus[valid], "LR+": lr_plus[valid],
        "TP": tp[valid], "FP": fp[valid], "FN": fn[valid], "TN": tn[valid],
        "TPR": tpr[valid], "FPR": fpr[valid], "threshold": thresholds[valid],
    }).sort_values(["LR-", "LR+"], ascending=[True, False]).reset_index(drop=True)

    if len(out):
        values = out["LR+"].to_numpy(dtype=float)
        prev = np.r_[-np.inf, np.maximum.accumulate(values)[:-1]]
        out = out.loc[values > prev + 1e-10].reset_index(drop=True)

    return out

def _fixedspec_metrics(tpr, fpr, prevalence):
    tpr = float(tpr); fpr = float(fpr); p = float(prevalence)
    positive_fraction = p * tpr + (1 - p) * fpr
    negative_fraction = p * (1 - tpr) + (1 - p) * (1 - fpr)
    return {
        "tpr": tpr,
        "ppv": p * tpr / positive_fraction if positive_fraction > 0 else np.nan,
        "npv": (1 - p) * (1 - fpr) / negative_fraction if negative_fraction > 0 else np.nan,
        "LR+": tpr / fpr if fpr > 0 else np.nan,
        "LR-": (1 - tpr) / (1 - fpr) if fpr < 1 else np.nan,
    }

def fixed_specificity_bootstrap(preds, *, specificity=0.95,
                                n_boot=5000, seed=0,
                                prevalence=None, confidence=0.95,
                                score_col="predicted_risk", label_col="target"):
    """Stratified score-level nonparametric ROC bootstrap at a fixed FPR.

    Percentile intervals are marginal pointwise 95% CIs. Each replicate
    reselects the score threshold implied by the desired specificity.
    """
    if not 0 < specificity < 1 or not 0 < confidence < 1 or n_boot < 200:
        raise ValueError("Invalid specificity, confidence or bootstrap count")
    y = pd.to_numeric(preds[label_col], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(preds[score_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(scores) & np.isin(y, (0, 1))
    y, scores = y[valid].astype(int), scores[valid]
    pos, neg = scores[y == 1], scores[y == 0]
    npos, nneg = len(pos), len(neg)
    if npos < 2 or nneg < 2:
        raise ValueError(f"Insufficient cases for bootstrap: {npos} positives, {nneg} negatives")
    fpr_target = 1 - float(specificity)
    if prevalence is None:
        p = npos / (npos + nneg)
        prevalence_is_fixed = False
    else:
        p = float(prevalence)
        prevalence_is_fixed = True
    if not 0 < p < 1:
        raise ValueError("Prevalence must be in (0,1)")
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    tpr_hat = float(np.interp(fpr_target, fpr, tpr))
    nominal = _fixedspec_metrics(tpr_hat, fpr_target, p)
    rng = np.random.default_rng(seed)
    samples = {metric: np.empty(n_boot, dtype=float) for metric in nominal}
    labels_b = np.r_[np.ones(npos, dtype=int), np.zeros(nneg, dtype=int)]
    for i in range(n_boot):
        scores_b = np.r_[pos[rng.integers(npos, size=npos)],
                         neg[rng.integers(nneg, size=nneg)]]
        fpr_b, tpr_b, _ = roc_curve(labels_b, scores_b, drop_intermediate=False)
        sens_b = float(np.interp(fpr_target, fpr_b, tpr_b))
        # Stratification conditions on case/control counts. If prevalence was
        # estimated from a representative cohort, simulate its independent
        # first-order binomial uncertainty for PPV and NPV.
        p_b = p if prevalence_is_fixed else rng.binomial(npos+nneg, p)/(npos+nneg)
        result_b = _fixedspec_metrics(sens_b, fpr_target, p_b)
        for metric in samples:
            samples[metric][i] = result_b[metric]
    tail = (1-confidence)/2
    result = {}
    for metric, value in nominal.items():
        vals = samples[metric]
        vals = vals[np.isfinite(vals)]
        if vals.size < 0.99*n_boot:
            raise ValueError(f"Too many non-finite bootstrap replicates for {metric}")
        lo, hi = np.quantile(vals, [tail, 1-tail])
        result[metric] = (float(value), float(lo), float(hi))
    return result, {"n_pos": npos, "n_neg": nneg, "prevalence": p,
                    "specificity": specificity, "n_boot": n_boot,
                    "prevalence_fixed": prevalence_is_fixed}

def score_curve_bundle(
    data,
    *,
    score_col="predicted_risk",
    label_col="target",
    prevalence=None,
    prevalence_sample_size=None,
    roc_alpha=0.05,
    roc_fpr_floor=0.001,
    roc_smoothing_step=0.01,
    roc_usample_precision=3,
    roc_min_expected_flags=10.0,
    roc_min_expected_fp=5.0,
    roc_min_expected_tn=5.0,
    roc_min_lrplus=1.001,
    pr_min_tp=10,
    pr_min_fp=5,
    pr_min_flags=20,
    pr_max_points=1000,
    pr_display_sigma=1.5,
    lr_min_cell_count=10,
    lr_fpr_min=0.005,
    lr_fpr_max=0.95,
    lr_min_plus=1.001,
    lr_display_sigma=4.0,
    confidence=0.95,
    allow_empty_lr=True,
):
    """Compute ROC, score-level PRC and supported LR frontier from scores.

    All parameters control reusable statistical/plot-geometry choices. Input
    DataFrame contains label and score columns; no dataset or path assumptions.
    Returns a dict with ``roc``, ``ppv`` and ``lr`` DataFrames. The ROC method
    follows the existing ``processRoc`` workflow, including its original
    smoothing, resampling and interval options; CI display smoothing is cosmetic.
    """
    if prevalence is None:
        prevalence = float(pd.to_numeric(data[label_col], errors="coerce").mean())
    if prevalence_sample_size is None:
        prevalence_sample_size = len(data)
    prevalence = float(prevalence)
    roc, n_total, n_pos = core.genroc(data, risk=score_col, target=label_col)
    zt = core.processRoc(
        roc, total_samples=n_total, positive_samples=n_pos,
        prevalence=prevalence, alpha=roc_alpha, lr_fpr_floor=roc_fpr_floor,
    )
    zt.set_lr_sp_floor(roc_fpr_floor)
    zt.smooth(STEP=roc_smoothing_step, convexify=False)
    zt.allmeasures(interpolate=True)
    zt.usample(precision=roc_usample_precision, recompute_measures=True)
    zt.getBounds(
        ppv_ci_method="direct", min_expected_flags=roc_min_expected_flags,
        min_expected_fp=roc_min_expected_fp, min_expected_tn=roc_min_expected_tn,
        enforce_bounds=True, smooth_ppv_bounds=True,
        mask_dominated_lr=True, lr_min_plus=roc_min_lrplus,
    )
    auc, auc_upper, auc_lower = zt.auc(bootstrap=False)
    curve = zt.get(bounds=True, interpolate=False)
    fpr = curve.index.to_numpy(dtype=float)
    tpr = curve["tpr"].to_numpy(dtype=float)
    flag_fraction = prevalence * tpr + (1 - prevalence) * fpr
    bad = (~np.isfinite(flag_fraction) | (n_total * flag_fraction < roc_min_expected_flags)
           | (flag_fraction <= 0))
    for col in ("ppv", "ppv_lower", "ppv_upper"):
        if col in curve:
            curve.loc[bad, col] = np.nan
    curve = curve.drop(columns="fpr", errors="ignore").reset_index()
    curve["AUC"], curve["AUC_upper"], curve["AUC_lower"] = auc, auc_upper, auc_lower

    pr = precision_recall_ci_from_scores(
        data, score_col=score_col, label_col=label_col,
        prevalence=prevalence, prevalence_sample_size=prevalence_sample_size,
        confidence=confidence, min_tp=pr_min_tp, min_fp=pr_min_fp,
        min_flags=pr_min_flags, max_points=pr_max_points,
        display_sigma=pr_display_sigma,
    )
    try:
        lr = lr_frontier_from_scores(
            data, score_col=score_col, label_col=label_col,
            min_count=lr_min_cell_count, fpr_min=lr_fpr_min,
            fpr_max=lr_fpr_max, min_lrplus=lr_min_plus,
        )
    except ValueError as exc:
        if not allow_empty_lr or not any(msg in str(exc) for msg in (
            "Too few positives or negatives", "No FPR search range supports"
        )):
            raise
        lr = pd.DataFrame(columns=["LR-", "LR+", "TP", "FP", "FN", "TN",
                                   "TPR", "FPR", "threshold"])
    lr = lr_logratio_ci(
        lr, confidence=confidence, min_count=lr_min_cell_count,
        smooth_sigma=lr_display_sigma,
    )
    return {"roc": curve, "ppv": pr, "lr": lr}


def plot_curve_panel(
    bundles,
    view,
    *,
    targets,
    cohorts,
    colors,
    cohort_styles,
    target_labels=None,
    output_stem=None,
    panel_side_in=1.85,
    margins_in=(0.52, 0.45, 0.18, 0.15),
    linewidth=1.2,
    png_dpi=400,
    pr_use_display=True,
    lr_use_display=True,
    lr_y_log_axis=False,
    roc_legend_position=(0.24, 0.02),
    show=False,
):
    """Plot the specified ROC, PPV or LR panel for arbitrary cohorts/targets.

    ``bundles`` maps ``(cohort, target)`` to score_curve_bundle output. The
    provided colors, styles and labels are presentation settings, not data
    model assumptions. When ``output_stem`` is set, saves PDF and PNG files.
    """
    view = view.upper()
    if view not in ("ROC", "PPV", "LR"):
        raise ValueError("view must be ROC, PPV, or LR")
    targets, cohorts = tuple(targets), tuple(cohorts)
    target_labels = dict(target_labels or {})
    left, bottom, right, top = map(float, margins_in)
    width = left + panel_side_in + right
    height = bottom + panel_side_in + top
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes([left / width, bottom / height,
                       panel_side_in / width, panel_side_in / height])
    names = {"ROC": ("fpr", "tpr", "1 - Specificity", "Sensitivity"),
             "PPV": ("tpr", "ppv", "Sensitivity", "PPV"),
             "LR": ("LR-", "LR+", "LR−", "LR+")}
    xcol, ycol, xlabel, ylabel = names[view]
    if view == "ROC":
        for cohort in cohorts:
            for target in targets:
                df = bundles[cohort, target]["roc"]
                x = df["fpr"].to_numpy(dtype=float)
                y = df["tpr"].to_numpy(dtype=float)
                lo = df["tpr_lower"].to_numpy(dtype=float)
                hi = df["tpr_upper"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                ax.fill_between(x, lo, hi, where=valid, color=colors[target],
                                alpha=cohort_styles[cohort]["alpha"], linewidth=0, zorder=1)
    elif view == "PPV":
        for cohort in cohorts:
            for target in targets:
                df = bundles[cohort, target]["ppv"]
                if not df.empty:
                    plot_precision_recall_ci_fill(
                        ax, df, color=colors[target], alpha=cohort_styles[cohort]["alpha"],
                        use_display=pr_use_display, zorder=1,
                    )
    handles = {}
    for cohort in cohorts:
        for target in targets:
            df = bundles[cohort, target][view.lower()]
            if df.empty:
                continue
            x, y = (df[c].to_numpy(dtype=float) for c in (xcol, ycol))
            valid = np.isfinite(x) & np.isfinite(y)
            if view == "LR":
                valid &= (x >= 0) & (y >= 0)
            line, = ax.plot(np.where(valid, x, np.nan), np.where(valid, y, np.nan),
                            color=colors[target], linestyle=cohort_styles[cohort]["linestyle"],
                            linewidth=linewidth, zorder=3)
            handles[cohort, target] = line
            if view == "LR":
                suffix = "_display" if lr_use_display else ""
                xx = df["LR-"].to_numpy(dtype=float)
                lo = df[f"LR+_logratio_lower{suffix}"].to_numpy(dtype=float)
                hi = df[f"LR+_logratio_upper{suffix}"].to_numpy(dtype=float)
                ok = np.isfinite(xx) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
                ax.fill_between(xx, lo, hi, where=ok, color=colors[target],
                                alpha=cohort_styles[cohort]["alpha"], linewidth=0,
                                interpolate=False, zorder=1)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, color="#808080", labelpad=2)
    ax.set_ylabel(ylabel, color="#808080", labelpad=3)
    ax.tick_params(which="both", length=0, colors="#808080", pad=2)
    for spine in ax.spines.values():
        spine.set_color("#F2F2F2")
        spine.set_linewidth(1.6)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#999999", alpha=0.5, linewidth=0.8)
    ax.grid(which="minor", color="#CCCCCC", alpha=0.4, linewidth=0.45)
    fmt = FuncFormatter(lambda x, pos: f"{x:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    if view in ("ROC", "PPV"):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    else:
        ax.margins(x=0, y=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        if lr_y_log_axis:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            from matplotlib.ticker import LogLocator
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if view == "ROC":
        legend_handles, labels = [], []
        for cohort in cohorts:
            legend_handles.append(Line2D([], [], linestyle="none"))
            labels.append(cohort)
            for target in targets:
                if (cohort, target) not in handles:
                    continue
                legend_handles.append(handles[cohort, target])
                df = bundles[cohort, target]["roc"]
                auc = float(df["AUC"].iloc[0])
                halfwidth = (float(df["AUC_upper"].iloc[0]) - float(df["AUC_lower"].iloc[0])) / 2
                labels.append(f"{target_labels.get(target, target)}: {100*auc:.1f}±{100*halfwidth:.1f}%")
        legend = ax.legend(legend_handles, labels, loc="lower left",
                           bbox_to_anchor=roc_legend_position, borderaxespad=0,
                           handlelength=2, handletextpad=0.5, labelspacing=0.14,
                           borderpad=0.15, frameon=True, facecolor="white",
                           edgecolor="none", framealpha=0.15,
                           prop={"size": 5.5, "weight": "semibold"})
        for text in legend.get_texts():
            text.set_color("black")
    if output_stem is not None:
        p = Path(output_stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p) + ".pdf", facecolor="white")
        fig.savefig(str(p) + ".png", dpi=png_dpi, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return (Path(str(output_stem) + ".pdf"), Path(str(output_stem) + ".png")) if output_stem is not None else None


def performance_table_latex(
    bundles,
    *,
    target_labels,
    cohorts,
    specificity,
    caption,
    label,
    bootstrap_estimates=None,
    operating_metrics=("tpr", "ppv", "npv", "LR+", "LR-"),
    footnotes=None,
):
    """Make a LaTeX operating-characteristics table from precomputed results.

    ``bundles`` maps ``(cohort, target)`` to score_curve_bundle output;
    ``bootstrap_estimates`` maps keys ``(cohort, target)`` to metric tuples
    ``(estimate, lower, upper)``. Cohorts, labels and caption are caller data.
    """
    target_labels = dict(target_labels)
    cohorts = tuple(cohorts)
    if not target_labels or not cohorts:
        raise ValueError("target_labels and cohorts must be nonempty")
    if not 0 < specificity < 1:
        raise ValueError("specificity must be in (0,1)")
    bootstrap_estimates = bootstrap_estimates or {}
    lines = [
        r"\begin{table}[ht!]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\sffamily\fontsize{9}{11}\selectfont",
        r"\resizebox{\textwidth}{!}{%", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.25}", r"\setlength{\extrarowheight}{2pt}",
        r"\begin{tabular}{L{0.75in}|L{0.95in}|r|r|r|r|r|r}",
        r"\hline", r"Dataset & Event & AUC & Sensitivity & PPV & NPV & LR+ & LR- \\",
        r"\hline",
    ]
    for cohort in cohorts:
        for index, (target, event_label) in enumerate(target_labels.items()):
            df = bundles[cohort, target]["roc"]
            row = df.iloc[np.abs(df["fpr"].to_numpy(dtype=float) - (1-specificity)).argmin()]
            metric_cells = []
            for metric in ("AUC", *operating_metrics):
                replacement = bootstrap_estimates.get((cohort, target))
                if metric != "AUC" and replacement is not None:
                    value, lower, upper = replacement[metric]
                else:
                    value = float(row[metric])
                    lower = float(row[metric + "_lower"])
                    upper = float(row[metric + "_upper"])
                if not np.isfinite(value):
                    cell = r"\textemdash"
                elif np.isfinite(lower) and np.isfinite(upper):
                    halfwidth = (upper-lower)/2
                    cell = rf"\mbox{{${value:.3f} \pm {halfwidth:.3f}$}}"
                else:
                    cell = rf"\mbox{{${value:.3f}$}}"
                metric_cells.append(cell)
            dataset_cell = (rf"\multirow{{{len(target_labels)}}}{{0.75in}}{{\raggedright {cohort}}}"
                            if index == 0 else "")
            lines.append(" & ".join([dataset_cell, event_label, *metric_cells]) + r" \\")
            lines.append(r"\hline" if index == len(target_labels)-1 else r"\cline{2-8}")
    lines.extend([r"\end{tabular}%", r"}", r"\vskip 1em",
                  r"\fontsize{9}{10}\selectfont", r"\raggedright"])
    if footnotes is None:
        footnotes = (
            r"$^\bigstar$ LR$+$: positive likelihood ratio, LR$-$: negative likelihood ratio, PPV: positive predictive value, NPV: negative predictive value.",
            rf"Operating metrics are evaluated at {100*specificity:g}\% specificity; intervals are pointwise and method-dependent.",
        )
    lines.extend(footnotes)
    lines.extend([r"\par", r"\end{table}"])
    return "\n".join(lines) + "\n"
