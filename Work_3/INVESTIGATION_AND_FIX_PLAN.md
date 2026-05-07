# iSFS BCD Investigation and Fix Plan

## User-Requested Items

1. Add GMEAN to the project metrics.
2. Add GMEAN to saved results, terminal outputs, Markdown reports, LaTeX reports, and generated PDF inputs.
3. Investigate whether the model trained and tested on the same full dataset.
4. Change evaluation to a train/test split, with a 70/30 split selected.
5. Verify the alpha/beta Block A loop against the multi-source BWM iSFS paper.
6. Verify the C update against the BCD/trust-region cost notes.
7. Verify the equations used for alpha, beta, BCE/logistic loss, class-cost loss, and trust-region radius.
8. Review `README.md`, `ISFS_CODE_PARAMETER_EXPLAINER.tex`, `ISFS_LOGISTIC_EXPLAINER.md`, `Multi_source_BWM.pdf`, and `notes.tex`.
9. Find additional issues or improvement areas in the current codebase.
10. Create a Markdown document in the repo so the investigation and task list are not lost.

## Investigation Findings

- GMEAN was missing from `compute_binary_metrics`, terminal metric output, profile CSVs, saved summary JSON, and generated Markdown/LaTeX reports.
- The old default path trained on `X_blocks, y` and evaluated on the same `X_blocks, y`, so reported performance was in-sample.
- The iSFS profile logic matches Section 3 of `Multi_source_BWM.pdf`: exact profiles are disjoint, while optimization groups overlap by source-combination containment.
- The alpha update matches the logistic extension of the constrained alpha subproblem: projected gradient on an l1 ball.
- The beta update matches the logistic extension of the regularized beta subproblem: proximal gradient with l1 soft-thresholding.
- The C update matches `notes.tex`: compute class-loss vector `L`, maximize the local linear model `L^T d`, clip by the working radius, global baseline ball, and nonnegativity, then accept/reject using actual vs predicted increase.
- Additional issue found: the active training path did not standardize features, while the paper assumes normalized columns and current source feature scales differ substantially.

## Implemented Tasks

- Added GMEAN, `sqrt(recall * specificity)`, to overall and profile metrics.
- Added a default 70/30 train/test split stratified by the joint key `(label, exact_missingness_profile)`.
- Added train-only feature standardization and imputation parameters, while preserving rows where an entire source is missing as all-NaN.
- Added explicit saved outputs for train and test summaries, exact-profile metrics, and optimization-group metrics.
- Kept backward-compatible aliases for `training_summary.json`, `exact_profile_metrics.csv`, and `optimization_group_metrics.csv`.
- Updated report generation to prefer split-aware outputs and fall back to old output folders.
- Added tests for GMEAN, stratified splitting, preprocessing mask preservation, and report rendering.

## Remaining Checks for Future Work

- Re-run full training with production iteration counts and regenerate the shareable report.
- If a publication-quality PDF is required, compile or post-process `results_report.tex`; the runner now also emits a compact PDF report directly.
- Consider adding hyperparameter tuning or cross-validation after the holdout path is stable.
