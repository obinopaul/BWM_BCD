"""
Generate compact markdown and LaTeX result tables from a saved training run.

Usage:
    python3 generate_results_report.py --run-dir training_outputs/real_20260424_010309

The script reads the saved JSON/CSV files produced by `run_training.py` and
creates:

    - results_report.md
    - results_report.tex

inside the selected run directory.
"""

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def format_float(value, digits=4):
    if value is None:
        return ""
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.{digits}f}"


def parse_vector(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    raise ValueError(f"Unsupported vector value: {value!r}")


def vector_to_text(values, digits=4):
    return "[" + ", ".join(format_float(v, digits=digits) for v in values) + "]"


def latex_escape(text):
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def alpha_rows(final_parameters):
    source_order = final_parameters["source_order"]
    rows = []
    for profile_key, weights in sorted(final_parameters["alphas"].items()):
        row = {
            "profile": profile_key.replace("|", ", "),
        }
        for source in source_order:
            row[source] = weights.get(source)
        rows.append(row)
    return rows


def beta_rows(final_parameters):
    rows = []
    for source, beta_values in final_parameters["betas"].items():
        beta = np.asarray(beta_values, dtype=float)
        rows.append(
            {
                "source": source,
                "n_features": int(beta.shape[0]),
                "l1_norm": float(np.sum(np.abs(beta))),
                "l2_norm": float(np.linalg.norm(beta)),
                "n_nonzero": int(np.sum(np.abs(beta) > 1e-12)),
            }
        )
    return rows


def metric_table_rows(summary):
    return [
        ("Accuracy", format_float(summary["accuracy"], 4)),
        ("Precision", format_float(summary["precision"], 4)),
        ("Recall / Sensitivity", format_float(summary["recall"], 4)),
        ("Specificity", format_float(summary["specificity"], 4)),
        ("F1 score", format_float(summary["f1"], 4)),
        ("Balanced accuracy", format_float(summary["balanced_accuracy"], 4)),
        ("Negative predictive value", format_float(summary["npv"], 4)),
        ("False positive rate", format_float(summary["false_positive_rate"], 4)),
        ("False negative rate", format_float(summary["false_negative_rate"], 4)),
        ("MCC", format_float(summary["matthews_corrcoef"], 4)),
        ("Brier score", format_float(summary["brier_score"], 4)),
        ("Log loss", format_float(summary["log_loss"], 4)),
    ]


def make_markdown(run_dir, config, summary, bcd_rows, exact_rows, opt_rows, final_parameters):
    final_costs = final_parameters["costs"]
    baseline_costs = final_parameters["baseline_costs"]
    alpha_table = alpha_rows(final_parameters)
    beta_table = beta_rows(final_parameters)
    accepted_steps = sum(row["accepted"] == "True" for row in bcd_rows)

    lines = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append("Brief note: training results for one saved iSFS logistic run on the real dataset.")
    lines.append("")
    lines.append("## Run Setup")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Dataset | `{config['dataset']}` |")
    lines.append(f"| Outer max iter | {config['max_iter']} |")
    lines.append(f"| Block A max iter | {config['block_a_max_iter']} |")
    lines.append(
        f"| Alpha/Beta subproblem max iter | {config['alpha_subproblem_max_iter']} / {config['beta_subproblem_max_iter']} |"
    )
    lines.append(f"| Learning rate alpha | {config['learning_rate_alpha']} |")
    lines.append(f"| Learning rate beta | {config['learning_rate_beta']} |")
    lines.append(f"| Lambda beta | {config['lambda_beta']} |")
    lines.append(f"| Alpha l1 radius | {config['alpha_l1_radius']} |")
    lines.append(f"| Decision threshold | {config['decision_threshold']} |")
    lines.append(f"| Initial cost vector | `{config['initial_cost_vector']}` |")
    lines.append(f"| Baseline cost vector | `{config['baseline_cost_vector']}` |")
    lines.append(f"| Final learned cost vector | `{vector_to_text(final_costs)}` |")
    lines.append("")

    lines.append("## Overall Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Samples | {summary['n_total_samples']} |")
    lines.append(f"| Valid predictions | {summary['n_valid_predictions']} |")
    lines.append(f"| TP | {summary['tp']} |")
    lines.append(f"| TN | {summary['tn']} |")
    lines.append(f"| FP | {summary['fp']} |")
    lines.append(f"| FN | {summary['fn']} |")
    for label, value in metric_table_rows(summary):
        lines.append(f"| {label} | {value} |")
    lines.append(
        f"| Probability range | `{format_float(summary['probability_min'], 4)} to {format_float(summary['probability_max'], 4)}` |"
    )
    lines.append("")

    lines.append("## BCD Summary")
    lines.append("")
    lines.append("| Recorded iterations | Accepted cost steps | Final cost vector |")
    lines.append("|---:|---:|---|")
    lines.append(f"| {len(bcd_rows)} | {accepted_steps} | `{vector_to_text(final_costs)}` |")
    lines.append("")
    lines.append("| Iter | Accepted | Pred. increase | Actual increase | Ratio | Radius | Current objective | Trial objective |")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for row in bcd_rows:
        lines.append(
            "| "
            f"{row['iteration']} | "
            f"{row['accepted']} | "
            f"{format_float(to_float(row['predicted_increase']), 4)} | "
            f"{format_float(to_float(row['actual_increase']), 4)} | "
            f"{format_float(to_float(row['ratio']), 4)} | "
            f"{format_float(to_float(row['working_radius']), 4)} | "
            f"{format_float(to_float(row['reduced_objective']), 4)} | "
            f"{format_float(to_float(row.get('trial_reduced_objective') or row['reduced_objective']), 4)} |"
        )
    lines.append("")

    lines.append("## Learned Alpha Weights")
    lines.append("")
    source_order = final_parameters["source_order"]
    header = "| Profile | " + " | ".join(source_order) + " |"
    divider = "|---|" + "|".join(["---:"] * len(source_order)) + "|"
    lines.append(header)
    lines.append(divider)
    for row in alpha_table:
        values = [format_float(row[source], 4) if row[source] is not None else "" for source in source_order]
        lines.append("| " + row["profile"] + " | " + " | ".join(values) + " |")
    lines.append("")

    lines.append("## Beta Summary")
    lines.append("")
    lines.append("| Source | Features | L1 norm | L2 norm | Nonzero coeffs |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in beta_table:
        lines.append(
            f"| {row['source']} | {row['n_features']} | {format_float(row['l1_norm'], 4)} | "
            f"{format_float(row['l2_norm'], 4)} | {row['n_nonzero']} |"
        )
    lines.append("")

    lines.append("## Exact Profile Results")
    lines.append("")
    lines.append("| Code | Samples | Accuracy | Precision | Recall | Specificity | F1 | BCE | Cost-BCE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in exact_rows:
        lines.append(
            f"| {row['profile_code']} | {row['n_samples']} | {format_float(row['accuracy'], 4)} | "
            f"{format_float(row['precision'], 4)} | {format_float(row['recall'], 4)} | "
            f"{format_float(row['specificity'], 4)} | {format_float(row['f1'], 4)} | "
            f"{format_float(row['bce'], 4)} | {format_float(row['cost_bce'], 4)} |"
        )
    lines.append("")

    lines.append("## Optimization Group Results")
    lines.append("")
    lines.append("| Code | Samples | Accuracy | Precision | Recall | Specificity | F1 | BCE | Cost-BCE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in opt_rows:
        lines.append(
            f"| {row['profile_code']} | {row['n_samples']} | {format_float(row['accuracy'], 4)} | "
            f"{format_float(row['precision'], 4)} | {format_float(row['recall'], 4)} | "
            f"{format_float(row['specificity'], 4)} | {format_float(row['f1'], 4)} | "
            f"{format_float(row['bce'], 4)} | {format_float(row['cost_bce'], 4)} |"
        )
    lines.append("")

    return "\n".join(lines)


def latex_table_overall(summary):
    rows = [
        ("Samples", summary["n_total_samples"]),
        ("Valid predictions", summary["n_valid_predictions"]),
        ("TP", summary["tp"]),
        ("TN", summary["tn"]),
        ("FP", summary["fp"]),
        ("FN", summary["fn"]),
    ]
    rows.extend(metric_table_rows(summary))
    rows.append(
        (
            "Probability range",
            f"{format_float(summary['probability_min'], 4)} to {format_float(summary['probability_max'], 4)}",
        )
    )
    body = "\n".join(
        f"{latex_escape(label)} & {latex_escape(value)} \\\\"
        for label, value in rows
    )
    return body


def latex_profile_rows(rows):
    body_lines = []
    for row in rows:
        body_lines.append(
            f"{latex_escape(row['profile_code'])} & "
            f"{int(float(row['n_samples']))} & "
            f"{format_float(float(row['accuracy']), 4)} & "
            f"{format_float(float(row['precision']), 4)} & "
            f"{format_float(float(row['recall']), 4)} & "
            f"{format_float(float(row['specificity']), 4)} & "
            f"{format_float(float(row['f1']), 4)} & "
            f"{format_float(float(row['bce']), 4)} & "
            f"{format_float(float(row['cost_bce']), 4)} \\\\"
        )
    return "\n".join(body_lines)


def make_latex(run_dir, config, summary, bcd_rows, exact_rows, opt_rows, final_parameters):
    source_order = final_parameters["source_order"]
    alpha_table = alpha_rows(final_parameters)
    beta_table = beta_rows(final_parameters)
    final_costs = vector_to_text(final_parameters["costs"])
    accepted_steps = sum(row["accepted"] == "True" for row in bcd_rows)

    alpha_header = "Profile & " + " & ".join(latex_escape(source) for source in source_order) + r" \\"
    alpha_rows_tex = []
    for row in alpha_table:
        values = [
            format_float(row[source], 4) if row[source] is not None else "-"
            for source in source_order
        ]
        alpha_rows_tex.append(
            latex_escape(row["profile"]) + " & " + " & ".join(values) + r" \\"
        )

    beta_rows_tex = []
    for row in beta_table:
        beta_rows_tex.append(
            f"{latex_escape(row['source'])} & {row['n_features']} & "
            f"{format_float(row['l1_norm'], 4)} & {format_float(row['l2_norm'], 4)} & "
            f"{row['n_nonzero']} \\\\"
        )

    bcd_rows_tex = []
    for row in bcd_rows:
        bcd_rows_tex.append(
            f"{row['iteration']} & {latex_escape(row['accepted'])} & "
            f"{format_float(to_float(row['predicted_increase']), 4)} & "
            f"{format_float(to_float(row['actual_increase']), 4)} & "
            f"{format_float(to_float(row['ratio']), 4)} & "
            f"{format_float(to_float(row['working_radius']), 4)} & "
            f"{format_float(to_float(row['reduced_objective']), 4)} \\\\"
        )

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,longtable,array}}
\usepackage{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6em}}
\begin{{document}}

\section*{{Results Summary}}

Brief note: training results for one saved iSFS logistic run on the real dataset.

\subsection*{{Run Setup}}
\begin{{tabular}}{{ll}}
\toprule
Item & Value \\
\midrule
Dataset & {latex_escape(config['dataset'])} \\
Outer max iter & {config['max_iter']} \\
Block A max iter & {config['block_a_max_iter']} \\
Alpha/Beta subproblem max iter & {config['alpha_subproblem_max_iter']} / {config['beta_subproblem_max_iter']} \\
Learning rate alpha & {config['learning_rate_alpha']} \\
Learning rate beta & {config['learning_rate_beta']} \\
Lambda beta & {config['lambda_beta']} \\
Alpha l1 radius & {config['alpha_l1_radius']} \\
Decision threshold & {config['decision_threshold']} \\
Initial cost vector & {latex_escape(config['initial_cost_vector'])} \\
Baseline cost vector & {latex_escape(config['baseline_cost_vector'])} \\
Final learned cost vector & {latex_escape(final_costs)} \\
\bottomrule
\end{{tabular}}

\subsection*{{Overall Performance}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
{latex_table_overall(summary)}
\bottomrule
\end{{tabular}}

\subsection*{{BCD Summary}}
\begin{{tabular}}{{lll}}
\toprule
Recorded iterations & Accepted cost steps & Final cost vector \\
\midrule
{len(bcd_rows)} & {accepted_steps} & {latex_escape(final_costs)} \\
\bottomrule
\end{{tabular}}

\begin{{center}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
Iter & Accept & Pred. inc & Actual inc & Ratio & Radius & Current obj \\
\midrule
{"".join(line + "\n" for line in bcd_rows_tex)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Learned Alpha Weights}}
\begin{{center}}
\begin{{tabular}}{{l{"r" * len(source_order)}}}
\toprule
{alpha_header}
\midrule
{"".join(line + "\n" for line in alpha_rows_tex)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Beta Summary}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
Source & Features & L1 norm & L2 norm & Nonzero coeffs \\
\midrule
{"".join(line + "\n" for line in beta_rows_tex)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Exact Profile Results}}
\begin{{center}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Code & Samples & Accuracy & Precision & Recall & Specificity & F1 & BCE & Cost-BCE \\
\midrule
{latex_profile_rows(exact_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Optimization Group Results}}
\begin{{center}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Code & Samples & Accuracy & Precision & Recall & Specificity & F1 & BCE & Cost-BCE \\
\midrule
{latex_profile_rows(opt_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\end{{document}}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown and LaTeX result tables from a saved training run."
    )
    parser.add_argument("--run-dir", required=True, help="Path to one saved training run directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "training_summary.json")
    final_parameters = load_json(run_dir / "final_parameters.json")
    bcd_rows = load_csv(run_dir / "bcd_history.csv")
    exact_rows = load_csv(run_dir / "exact_profile_metrics.csv")
    opt_rows = load_csv(run_dir / "optimization_group_metrics.csv")

    markdown = make_markdown(run_dir, config, summary, bcd_rows, exact_rows, opt_rows, final_parameters)
    latex = make_latex(run_dir, config, summary, bcd_rows, exact_rows, opt_rows, final_parameters)

    md_path = run_dir / "results_report.md"
    tex_path = run_dir / "results_report.tex"
    md_path.write_text(markdown, encoding="utf-8")
    tex_path.write_text(latex, encoding="utf-8")

    print(f"Generated: {md_path}")
    print(f"Generated: {tex_path}")


if __name__ == "__main__":
    main()
