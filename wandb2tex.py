#!/usr/bin/env python3
import argparse, json, sys, math, re
from typing import List, Dict, Any, Optional

import pandas as pd
from requests.exceptions import HTTPError
import wandb

def fetch_runs(entity: str, project: str, sweep: Optional[str], filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    api = wandb.Api()
    # filters = {"summary.fft_feat_alpha": {"$eq": 0.3},
    #            "summary.fft_feat_keep_ratio":{"$eq": 0.8},
    #            "summary.fft_input_alpha":{"$eq": 0.5},
    #            "summary.fft_input_keep_ratio":{"$eq": 0.6}}

    filters = {
               "config.fft_input_alpha":0.5,
                "config.fft_input_keep_ratio":0.6,
                "config.fft_feat_alpha": 0.3,
                "config.fft_feat_keep_ratio":0.8
                }
    # filters = {'config.fft_feat_alpha': 0.3}
    try:
        if sweep:
            sw = api.sweep(f"{entity}/{project}/{sweep}")
            runs = list(sw.runs())
        else:
            runs = list(api.runs(f"{entity}/{project}", filters=filters or {}))
    except HTTPError as exc:
        message = "Failed to fetch runs from W&B (HTTP error)."
        if filters:
            message += " Check that --filters uses valid W&B syntax."
        raise RuntimeError(message) from exc

    # Flatten config + summary into columns
    rows = []
    for r in runs:
        row = {
            "run_id": r.id,
            "name": r.name,
            "state": r.state,
            "tags": ",".join(r.tags or []),
            "group": r.group,
        }
        # summary metrics (finals)
        for k, v in (r.summary or {}).items():
            if isinstance(v, (int, float, str)):
                row[f"summary.{k}"] = v
        # config (hyperparams)
        for k, v in (r.config or {}).items():
            if isinstance(v, (int, float, str, bool)):
                row[f"cfg.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)

def pivot_df(df: pd.DataFrame, index_cols: List[str], column_col: str, value_cols: List[str], agg: str):
    # melt then pivot for multiple values
    sub = df[index_cols + [column_col] + value_cols].copy()
    melted = sub.melt(id_vars=index_cols + [column_col], value_vars=value_cols,
                      var_name="metric", value_name="value")

    # build pivot with a multi-index so we can control the ordering
    piv = melted.pivot_table(
        index=index_cols,
        columns=[column_col, "metric"],
        values="value",
        aggfunc=agg
    )

    # ensure attack rates (column_col values) are sorted numerically when possible
    def sort_key(val: Any):
        if pd.isna(val):
            return (2, "")
        try:
            return (0, float(val))
        except (TypeError, ValueError):
            return (1, str(val))

    col_values_raw = list(pd.unique(melted[column_col]))
    col_values = sorted(col_values_raw, key=sort_key)

    metric_values = list(pd.unique(melted["metric"]))
    metrics_present = [m for m in value_cols if m in metric_values]

    desired_cols = pd.MultiIndex.from_product(
        [col_values, metrics_present],
        names=[column_col, "metric"]
    )
    piv = piv.reindex(columns=desired_cols)

    # flatten multi-index columns back to "metric|attack" strings
    flat_cols = [f"{metric}|{str(col_val)}" for col_val, metric in piv.columns]
    piv.columns = flat_cols

    piv = piv.reset_index()
    return piv

def bold_best(df: pd.DataFrame, higher_is_better: bool = True, per: str = "column", eps: float = 1e-12) -> pd.DataFrame:
    formatted = df.copy()
    if per == "column":
        start = 0
        for col in formatted.columns:
            if pd.api.types.is_numeric_dtype(formatted[col]):
                series = formatted[col]
                if higher_is_better:
                    m = series.max()
                    mask = series >= (m - eps)
                else:
                    m = series.min()
                    mask = series <= (m + eps)
                formatted[col] = [f"\\textbf{{{x:.2f}}}" if ok and pd.notnull(x) else (f"{x:.2f}" if pd.notnull(x) else "") for x, ok in zip(series, mask)]
    else:
        raise NotImplementedError
    return formatted

def format_numeric(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else "")
    return out

def main():
    p = argparse.ArgumentParser(description="Export W&B runs to LaTeX tables")
    p.add_argument("--entity", default="bigslav")
    p.add_argument("--project", default='fft_runs')
    p.add_argument("--sweep", default=None, help="Optional sweep id")
    p.add_argument("--filters", default='',
        help="JSON dict for Api.runs filters (e.g. '{\"config.param\": 1}')"
    )
    p.add_argument("--index", nargs="+", default=["cfg.dataset","cfg.test_envs", "cfg.adapt_alg"], help="Index columns for rows")
    p.add_argument("--column", default="cfg.attack_rate", help="Column name to spread across columns")
    p.add_argument("--values", nargs="+", default=["summary.acc_mean", "summary.acc_std"], help="Value columns (summary metrics)")
    p.add_argument("--agg", default="sum", choices=["mean","max","min", "sum"])
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--bold-best", action="store_true")
    p.add_argument("--higher-is-better", action="store_true", help="Use with --bold-best")
    p.add_argument("--caption", default="Results")
    p.add_argument("--label", default="tab:results")
    p.add_argument("--longtable", action="store_true")
    p.add_argument("--siunitx", action="store_true", help="Use siunitx S columns")
    p.add_argument("--outfile", default=None, help="Write LaTeX to file; default prints to stdout")
    args = p.parse_args()

    if args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON passed via --filters: {exc}")
    else:
        filters = None
    df = fetch_runs(args.entity, args.project, args.sweep, filters)
    if df.empty:
        print("% No runs matched.", file=sys.stderr)
        sys.exit(2)

    piv = pivot_df(df, args.index, args.column, args.values, args.agg)
    piv = format_numeric(piv, args.decimals)

    if args.bold_best:
        # Apply only to metric columns (not index)
        metric_cols = [c for c in piv.columns if "|" in c]
        styled = piv.copy()
        styled[metric_cols] = bold_best(piv[metric_cols], higher_is_better=args.higher_is_better)
        piv = styled

    # Build LaTeX
    col_format = None
    if args.siunitx:
        # one l for index cols, then S for numeric-like columns (we've formatted strings, but journal comps still accept)
        n_idx = len(args.index)
        n_val = len([c for c in piv.columns if c not in args.index])
        col_format = "l" * n_idx + "S" * n_val

    latex = piv.to_latex(
        index=False,
        escape=True,          # allow \textbf
        longtable=args.longtable,
        column_format=col_format,
        caption=args.caption,
        label=args.label,
        bold_rows=False
    )  # pandas handles booktabs by default; add \usepackage{booktabs}

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(latex)
    else:
        print(latex)

if __name__ == "__main__":
    main()
