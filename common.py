from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import pandas as pd
from rich.text import Text
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SLICE_FEATURES = [
    "donation_count",
    "total_amount",
    "avg_amount",
    "amount_std",
    "max_amount",
    "days_since_last_donation",
    "days_since_first_donation",
    "had_donation",
]

SLICE_DEBUG_RENAMES = {
    "slice_index": "sl_idx",
    "slice_start": "sl_start",
    "slice_end": "sl_end",
    "donation_count": "don_ct",
    "total_amount": "tot_amt",
    "avg_amount": "avg_amt",
    "amount_std": "amt_std",
    "max_amount": "max_amt",
    "days_since_last_donation": "dsl_don",
    "days_since_first_donation": "dsf_don",
    "had_donation": "had_don",
}

SLICE_DEBUG_APPENDIX = [
    ("sl_idx", "Slice index within lookback window"),
    ("sl_start", "Slice start date"),
    ("sl_end", "Slice end date"),
    ("don_ct", "Donation count in slice"),
    ("tot_amt", "Total donation amount in slice"),
    ("avg_amt", "Average donation amount in slice"),
    ("amt_std", "Donation amount standard deviation in slice"),
    ("max_amt", "Maximum donation amount in slice"),
    ("dsl_don", "Days since last donation as of slice end"),
    ("dsf_don", "Days since first donation as of slice end"),
    ("had_don", "1 if slice has any donation, else 0"),
]


@dataclass
class SequenceExperimentResult:
    overall_metrics: dict
    per_user_metrics: pd.DataFrame
    predictions: pd.DataFrame
    dataset: pd.DataFrame
    model_dir: Path


@dataclass
class PreparedSequenceData:
    dataset: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cache_dir: Path
    dataset_processing_seconds: float


def make_sequence_feature_names(lookback_slices: int) -> list[str]:
    names: list[str] = []
    for offset in range(lookback_slices):
        prefix = f"slice_{offset + 1}"
        for feature in SLICE_FEATURES:
            names.append(f"{prefix}_{feature}")
    return names


def _safe_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=0))


def load_donations(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["donation_date"] = pd.to_datetime(df["donation_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["email", "donation_date"])
    return df.sort_values(["email", "donation_date", "amount"]).reset_index(drop=True)


def _slice_features(
    user_history: pd.DataFrame,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
) -> dict[str, float]:
    history_until_end = user_history.loc[user_history["donation_date"] <= slice_end]
    in_slice = user_history.loc[
        (user_history["donation_date"] >= slice_start) & (user_history["donation_date"] <= slice_end)
    ]

    amounts = in_slice["amount"].astype(float)

    if history_until_end.empty:
        days_since_last = float((slice_end - slice_start).days) + 1.0
        days_since_first = days_since_last
    else:
        last_date = history_until_end["donation_date"].max()
        first_date = history_until_end["donation_date"].min()
        days_since_last = float((slice_end - last_date).days)
        days_since_first = float((slice_end - first_date).days)

    return {
        "donation_count": float(len(in_slice)),
        "total_amount": float(amounts.sum()) if not in_slice.empty else 0.0,
        "avg_amount": float(amounts.mean()) if not in_slice.empty else 0.0,
        "amount_std": _safe_std(amounts) if not in_slice.empty else 0.0,
        "max_amount": float(amounts.max()) if not in_slice.empty else 0.0,
        "days_since_last_donation": days_since_last,
        "days_since_first_donation": days_since_first,
        "had_donation": float(not in_slice.empty),
    }


def build_sequence_example(
    user_history: pd.DataFrame,
    anchor_date: pd.Timestamp,
    slice_days: int,
    lookback_slices: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    feature_row: dict[str, float] = {}
    debug_rows: list[dict] = []

    for offset in range(lookback_slices):
        step = lookback_slices - offset - 1
        slice_end = anchor_date - pd.Timedelta(days=step * slice_days)
        slice_start = slice_end - pd.Timedelta(days=slice_days - 1)
        features = _slice_features(user_history, slice_start, slice_end)
        prefix = f"slice_{offset + 1}"
        for name, value in features.items():
            feature_row[f"{prefix}_{name}"] = value
        debug_rows.append(
            {
                "slice_index": offset + 1,
                "slice_start": slice_start,
                "slice_end": slice_end,
                **features,
            }
        )

    return feature_row, pd.DataFrame(debug_rows)


def build_sequence_dataset(
    donations: pd.DataFrame,
    horizon_days: int = 90,
    slice_days: int = 30,
    lookback_slices: int = 6,
    min_examples_per_user: int = 2,
    anchor_stride_days: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    max_observed_date = donations["donation_date"].max()
    latest_anchor = max_observed_date - pd.Timedelta(days=horizon_days)
    feature_names = make_sequence_feature_names(lookback_slices)

    stride_days = anchor_stride_days or slice_days
    for email, group in donations.groupby("email", sort=False):
        user_history = group.sort_values("donation_date").reset_index(drop=True)
        if user_history.empty:
            continue

        first_possible_anchor = user_history["donation_date"].min() + pd.Timedelta(days=(lookback_slices - 1) * slice_days)
        anchor_dates = pd.date_range(start=first_possible_anchor, end=latest_anchor, freq=f"{stride_days}D")
        user_rows = 0
        for anchor_date in anchor_dates:
            feature_row, _ = build_sequence_example(user_history, anchor_date, slice_days, lookback_slices)
            future_window = user_history.loc[
                (user_history["donation_date"] > anchor_date)
                & (user_history["donation_date"] <= anchor_date + pd.Timedelta(days=horizon_days))
            ]
            row = {
                "email": email,
                "anchor_date": anchor_date,
                "label": int(not future_window.empty),
            }
            row.update(feature_row)
            rows.append(row)
            user_rows += 1

        if user_rows < min_examples_per_user:
            rows = rows[:-user_rows]

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset
    return dataset.sort_values(["email", "anchor_date"]).reset_index(drop=True)


def split_sequence_dataset(
    dataset: pd.DataFrame,
    lookback_slices: int,
    test_size: float = 0.25,
    min_train_examples: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for _, group in dataset.groupby("email", sort=False):
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        test_n = max(1, math.ceil(len(group) * test_size))
        train_n = len(group) - test_n
        if train_n < min_train_examples:
            continue
        train_frames.append(group.iloc[:train_n].copy())
        test_frames.append(group.iloc[train_n:].copy())

    if not train_frames or not test_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def _prepared_cache_dir(
    output_root: str | Path,
    horizon_days: int,
    slice_days: int,
    lookback_slices: int,
    min_examples_per_user: int,
    anchor_stride_days: int | None,
    test_size: float,
    random_state: int,
) -> Path:
    stride_days = anchor_stride_days or slice_days
    key = (
        f"h{horizon_days}_s{slice_days}_l{lookback_slices}_"
        f"stride{stride_days}_min{min_examples_per_user}_test{str(test_size).replace('.', 'p')}_"
        f"seed{random_state}"
    )
    return Path(output_root) / "_prepared_sequences" / key


def load_or_prepare_sequence_data(
    xlsx_path: str | Path,
    output_root: str | Path,
    horizon_days: int,
    test_size: float,
    slice_days: int,
    lookback_slices: int,
    min_examples_per_user: int,
    anchor_stride_days: int | None,
    random_state: int,
) -> PreparedSequenceData:
    cache_dir = _prepared_cache_dir(
        output_root=output_root,
        horizon_days=horizon_days,
        slice_days=slice_days,
        lookback_slices=lookback_slices,
        min_examples_per_user=min_examples_per_user,
        anchor_stride_days=anchor_stride_days,
        test_size=test_size,
        random_state=random_state,
    )
    dataset_path = cache_dir / "sequence_dataset.csv"
    train_path = cache_dir / "train_sequence_dataset.csv"
    test_path = cache_dir / "test_sequence_dataset.csv"
    metadata_path = cache_dir / "prepared_metadata.json"

    if dataset_path.exists() and train_path.exists() and test_path.exists() and metadata_path.exists():
        print(f"[dataset-cache] Reusing prepared dataset from {cache_dir}")
        metadata = json.loads(metadata_path.read_text())
        return PreparedSequenceData(
            dataset=pd.read_csv(dataset_path, parse_dates=["anchor_date"]),
            train_df=pd.read_csv(train_path, parse_dates=["anchor_date"]),
            test_df=pd.read_csv(test_path, parse_dates=["anchor_date"]),
            cache_dir=cache_dir,
            dataset_processing_seconds=float(metadata.get("dataset_processing_seconds", 0.0)),
        )

    print(f"[dataset-cache] Building prepared dataset at {cache_dir}")
    started_at = time.perf_counter()
    donations = load_donations(xlsx_path)
    dataset = build_sequence_dataset(
        donations=donations,
        horizon_days=horizon_days,
        slice_days=slice_days,
        lookback_slices=lookback_slices,
        min_examples_per_user=min_examples_per_user,
        anchor_stride_days=anchor_stride_days,
    )
    train_df, test_df = split_sequence_dataset(
        dataset=dataset,
        lookback_slices=lookback_slices,
        test_size=test_size,
        min_train_examples=min_examples_per_user,
        random_state=random_state,
    )
    dataset_processing_seconds = time.perf_counter() - started_at

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    save_json(
        metadata_path,
        {
            "horizon_days": horizon_days,
            "slice_days": slice_days,
            "lookback_slices": lookback_slices,
            "anchor_stride_days": anchor_stride_days or slice_days,
            "min_examples_per_user": min_examples_per_user,
            "test_size": test_size,
            "random_state": random_state,
            "dataset_processing_seconds": dataset_processing_seconds,
        },
    )
    return PreparedSequenceData(
        dataset=dataset,
        train_df=train_df,
        test_df=test_df,
        cache_dir=cache_dir,
        dataset_processing_seconds=dataset_processing_seconds,
    )


def reshape_flat_features_to_sequence(frame: pd.DataFrame, lookback_slices: int) -> np.ndarray:
    feature_names = make_sequence_feature_names(lookback_slices)
    array = frame[feature_names].to_numpy(dtype=np.float32)
    return array.reshape(len(frame), lookback_slices, len(SLICE_FEATURES))


def build_numeric_preprocessor(scale: bool, feature_names: list[str]) -> ColumnTransformer:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), feature_names)],
        remainder="drop",
    )


def resolve_tabular_normalization(normalize_requested: bool) -> tuple[bool, str]:
    apply_normalization = bool(normalize_requested)
    method = "standard_scaler" if apply_normalization else "none"
    return apply_normalization, method


def resolve_torch_normalization(normalize_requested: bool) -> tuple[bool, str]:
    apply_normalization = bool(normalize_requested)
    method = "trainset_zscore" if apply_normalization else "none"
    return apply_normalization, method


def infer_saved_tabular_normalization(metadata: dict) -> tuple[bool, str]:
    if "normalize" in metadata:
        apply_normalization = bool(metadata["normalize"])
        return apply_normalization, str(metadata.get("normalization_method", "standard_scaler" if apply_normalization else "none"))
    # Backward compatibility for older saved artifacts.
    model_name = str(metadata.get("model_name", ""))
    apply_normalization = model_name == "logistic_regression"
    method = "standard_scaler" if apply_normalization else "none"
    return apply_normalization, method


def infer_saved_torch_normalization(metadata: dict, checkpoint: dict) -> tuple[bool, str]:
    if "normalize" in metadata:
        apply_normalization = bool(metadata["normalize"])
        return apply_normalization, str(metadata.get("normalization_method", "trainset_zscore" if apply_normalization else "none"))
    if "normalize" in checkpoint:
        apply_normalization = bool(checkpoint["normalize"])
        return apply_normalization, str(checkpoint.get("normalization_method", "trainset_zscore" if apply_normalization else "none"))
    # Backward compatibility for older torch checkpoints, which always normalized.
    return True, "trainset_zscore"


def summarize_user_history_stats(user_history: pd.DataFrame) -> pd.DataFrame:
    amounts = user_history["amount"].astype(float)
    donation_dates = pd.to_datetime(user_history["donation_date"], errors="coerce").sort_values()
    gap_days = donation_dates.diff().dt.days.dropna().astype(float)

    rows = [
        {
            "metric": "donation_amount",
            "count": int(len(amounts)),
            "mean": float(amounts.mean()) if not amounts.empty else 0.0,
            "std": _safe_std(amounts) if not amounts.empty else 0.0,
            "min": float(amounts.min()) if not amounts.empty else 0.0,
            "median": float(amounts.median()) if not amounts.empty else 0.0,
            "max": float(amounts.max()) if not amounts.empty else 0.0,
        },
        {
            "metric": "donation_gap_days",
            "count": int(len(gap_days)),
            "mean": float(gap_days.mean()) if not gap_days.empty else 0.0,
            "std": _safe_std(gap_days) if not gap_days.empty else 0.0,
            "min": float(gap_days.min()) if not gap_days.empty else 0.0,
            "median": float(gap_days.median()) if not gap_days.empty else 0.0,
            "max": float(gap_days.max()) if not gap_days.empty else 0.0,
        },
    ]
    return pd.DataFrame(rows)


def safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_balanced_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(balanced_accuracy_score(y_true, y_pred))


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "roc_auc": safe_roc_auc(y_true, y_score),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_balanced_accuracy(y_true, y_pred),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - y_score, y_score]), labels=[0, 1])),
    }


def evaluate_by_user(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for email, group in predictions.groupby("email", sort=False):
        y_true = group["label"].astype(int)
        y_score = group["score"].astype(float).to_numpy()
        y_pred = group["predicted_label"].astype(int).to_numpy()
        rows.append(
            {
                "email": email,
                "n_samples": int(len(group)),
                "positive_rate": float(y_true.mean()),
                "mean_score": float(group["score"].mean()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": safe_balanced_accuracy(y_true, y_pred),
                "brier_score": float(brier_score_loss(y_true, y_score)),
                "log_loss": float(log_loss(y_true, np.column_stack([1 - y_score, y_score]), labels=[0, 1])),
                "roc_auc": safe_roc_auc(y_true, y_score),
            }
        )
    return pd.DataFrame(rows).sort_values(["brier_score", "email"]).reset_index(drop=True)


def build_balanced_eval_subsets(
    donations: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_days: int,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    if test_df.empty:
        return {f"future_{count}_donations": pd.DataFrame() for count in [1, 2, 3]}

    horizon_delta = pd.Timedelta(days=horizon_days)
    donation_dates_by_email = {
        email: group["donation_date"].sort_values().to_numpy()
        for email, group in donations.groupby("email", sort=False)
    }

    future_counts: list[int] = []
    for row in test_df.itertuples(index=False):
        donation_dates = donation_dates_by_email.get(row.email)
        if donation_dates is None:
            future_counts.append(0)
            continue
        future_count = int(((donation_dates > row.anchor_date) & (donation_dates <= row.anchor_date + horizon_delta)).sum())
        future_counts.append(future_count)

    eval_base = test_df.copy()
    eval_base["future_donation_count"] = future_counts
    negatives = eval_base.loc[eval_base["label"] == 0].copy()

    subsets: dict[str, pd.DataFrame] = {}
    for donation_count in [1, 2, 3]:
        subset_name = f"future_{donation_count}_donations"
        positives = eval_base.loc[
            (eval_base["label"] == 1) & (eval_base["future_donation_count"] == donation_count)
        ].copy()
        sample_n = min(len(positives), len(negatives))
        if sample_n == 0:
            subsets[subset_name] = pd.DataFrame(columns=eval_base.columns)
            continue
        positive_sample = positives.sample(n=sample_n, random_state=random_state + donation_count)
        negative_sample = negatives.sample(
            n=sample_n,
            replace=True,
            random_state=random_state + 100 + donation_count,
        ).copy()
        subset = pd.concat([positive_sample, negative_sample], ignore_index=True)
        subset = subset.sample(frac=1.0, random_state=random_state + 200 + donation_count).reset_index(drop=True)
        subsets[subset_name] = subset
    return subsets


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def print_eval_balance_concat_debug(model_name: str, per_user_frames: list[pd.DataFrame]) -> None:
    console = Console()
    debug_table = Table(title=f"Eval-Balance Concat Debug: {model_name}", show_lines=False)
    debug_table.add_column("frame_idx")
    debug_table.add_column("eval_subset")
    debug_table.add_column("rows")
    debug_table.add_column("empty")
    debug_table.add_column("all_na_columns")

    for idx, frame in enumerate(per_user_frames, start=1):
        subset_name = (
            str(frame["eval_subset"].iloc[0])
            if not frame.empty and "eval_subset" in frame.columns
            else "unknown"
        )
        all_na_columns = [column for column in frame.columns if frame[column].isna().all()]
        debug_table.add_row(
            str(idx),
            subset_name,
            str(len(frame)),
            str(frame.empty),
            ", ".join(all_na_columns) if all_na_columns else "-",
        )
    console.print(debug_table)


def drop_all_na_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.dropna(axis=1, how="all")


def _build_pipeline(estimator_factory: Callable[[], object], scale_numeric: bool, feature_names: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_numeric_preprocessor(scale=scale_numeric, feature_names=feature_names)),
            ("classifier", estimator_factory()),
        ]
    )


def print_training_debug(
    model_name: str,
    donations: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_days: int,
    slice_days: int,
    lookback_slices: int,
    anchor_stride_days: int,
) -> None:
    console = Console()

    def _sample_stats(frame: pd.DataFrame) -> dict[str, float]:
        counts = frame.groupby("email").size()
        return {
            "users": int(frame["email"].nunique()),
            "samples": int(len(frame)),
            "min_per_user": int(counts.min()),
            "median_per_user": float(counts.median()),
            "mean_per_user": float(counts.mean()),
            "max_per_user": int(counts.max()),
        }

    def _history_span_stats(emails: pd.Series) -> dict[str, float]:
        included_emails = emails.drop_duplicates()
        spans = (
            donations.loc[donations["email"].isin(included_emails), ["email", "donation_date"]]
            .groupby("email")["donation_date"]
            .agg(["min", "max"])
        )
        history_lengths = (spans["max"] - spans["min"]).dt.days.astype(float)
        if history_lengths.empty:
            return {"mean_days": 0.0, "std_days": 0.0}
        return {
            "mean_days": float(history_lengths.mean()),
            "std_days": float(history_lengths.std(ddof=0)) if len(history_lengths) > 1 else 0.0,
        }

    train_stats = _sample_stats(train_df)
    test_stats = _sample_stats(test_df)
    train_history_span_stats = _history_span_stats(train_df["email"])
    test_history_span_stats = _history_span_stats(test_df["email"])

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Model:[/bold] {model_name}",
                    f"[bold]Horizon Days:[/bold] {horizon_days}",
                    f"[bold]Slice Days:[/bold] {slice_days}",
                    f"[bold]Lookback Slices:[/bold] {lookback_slices}",
                    f"[bold]Anchor Stride Days:[/bold] {anchor_stride_days}",
                    f"[bold]Included Train Users:[/bold] {train_stats['users']}",
                    f"[bold]Included Test Users:[/bold] {test_stats['users']}",
                    f"[bold]Train User History Length Mean (days):[/bold] {train_history_span_stats['mean_days']:.2f}",
                    f"[bold]Train User History Length Std (days):[/bold] {train_history_span_stats['std_days']:.2f}",
                    f"[bold]Test User History Length Mean (days):[/bold] {test_history_span_stats['mean_days']:.2f}",
                    f"[bold]Test User History Length Std (days):[/bold] {test_history_span_stats['std_days']:.2f}",
                ]
            ),
            title="Training Debug",
            border_style="magenta",
        )
    )

    stats_table = Table(title="Per-User Sample Statistics", show_lines=False)
    stats_table.add_column("split")
    stats_table.add_column("users")
    stats_table.add_column("samples")
    stats_table.add_column("min/user")
    stats_table.add_column("median/user")
    stats_table.add_column("mean/user")
    stats_table.add_column("max/user")
    stats_table.add_row(
        "train",
        str(train_stats["users"]),
        str(train_stats["samples"]),
        str(train_stats["min_per_user"]),
        f"{train_stats['median_per_user']:.2f}",
        f"{train_stats['mean_per_user']:.2f}",
        str(train_stats["max_per_user"]),
    )
    stats_table.add_row(
        "test",
        str(test_stats["users"]),
        str(test_stats["samples"]),
        str(test_stats["min_per_user"]),
        f"{test_stats['median_per_user']:.2f}",
        f"{test_stats['mean_per_user']:.2f}",
        str(test_stats["max_per_user"]),
    )
    console.print(stats_table)


def print_training_result(model_dir: Path, metrics: dict) -> None:
    console = Console()
    console.print(
        Panel.fit(
            f"[bold green]Saved outputs to[/bold green] {model_dir}",
            title="Training Output",
            border_style="green",
        )
    )

    metrics_table = Table(title="Training Metrics", show_lines=False)
    metrics_table.add_column("metric")
    metrics_table.add_column("value")
    skip_keys: set[str] = set()
    if metrics.get("eval_balance"):
        for subset_name in ["future_1_donations", "future_2_donations", "future_3_donations"]:
            sample_key = f"{subset_name}_n_samples"
            if sample_key in metrics:
                metrics_table.add_row(sample_key, str(metrics[sample_key]))
            for metric_name in [
                "roc_auc",
                "average_precision",
                "accuracy",
                "balanced_accuracy",
                "brier_score",
                "log_loss",
            ]:
                key = f"{subset_name}_{metric_name}"
                if key not in metrics:
                    continue
                value = metrics[key]
                label = f"{subset_name} {metric_name}"
                if isinstance(value, float):
                    metrics_table.add_row(label, f"{value:.6f}")
                else:
                    metrics_table.add_row(label, str(value))
                skip_keys.add(key)
            skip_keys.add(sample_key)
    for key, value in metrics.items():
        if key in skip_keys:
            continue
        if isinstance(value, float):
            metrics_table.add_row(key, f"{value:.6f}")
        else:
            metrics_table.add_row(key, str(value))
    console.print(metrics_table)


def save_torch_checkpoint(
    model_dir: Path,
    model_name: str,
    model: nn.Module,
    horizon_days: int,
    slice_days: int,
    lookback_slices: int,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    normalize: bool,
    normalization_method: str,
    extra_metadata: dict | None = None,
) -> Path:
    model_path = model_dir / "trained_model.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "horizon_days": horizon_days,
        "slice_days": slice_days,
        "lookback_slices": lookback_slices,
        "slice_features": SLICE_FEATURES,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "normalize": normalize,
        "normalization_method": normalization_method,
        "extra_metadata": extra_metadata or {},
    }
    torch.save(payload, model_path)
    return model_path


def run_sequence_experiment(
    model_name: str,
    estimator_factory: Callable[[], object],
    xlsx_path: str | Path,
    output_root: str | Path = "outputs",
    horizon_days: int = 90,
    test_size: float = 0.25,
    scale_numeric: bool = True,
    slice_days: int = 30,
    lookback_slices: int = 6,
    min_examples_per_user: int = 2,
    anchor_stride_days: int | None = None,
    class_imbalance: bool = False,
    random_state: int = 42,
    eval_balance: bool = False,
    normalize: bool = False,
) -> SequenceExperimentResult:
    prepared = load_or_prepare_sequence_data(
        xlsx_path=xlsx_path,
        output_root=output_root,
        horizon_days=horizon_days,
        test_size=test_size,
        slice_days=slice_days,
        lookback_slices=lookback_slices,
        min_examples_per_user=min_examples_per_user,
        anchor_stride_days=anchor_stride_days,
        random_state=random_state,
    )
    feature_names = make_sequence_feature_names(lookback_slices)
    dataset = prepared.dataset
    train_df = prepared.train_df
    test_df = prepared.test_df
    donations = load_donations(xlsx_path)
    if train_df.empty or test_df.empty:
        raise ValueError("No train/test sequence samples were created. Adjust slice_days/lookback_slices/horizon_days.")
    print_training_debug(
        model_name,
        donations,
        train_df,
        test_df,
        horizon_days,
        slice_days,
        lookback_slices,
        anchor_stride_days or slice_days,
    )

    fit_started_at = time.perf_counter()
    apply_normalization, normalization_method = resolve_tabular_normalization(normalize_requested=normalize)
    model = _build_pipeline(estimator_factory, scale_numeric=apply_normalization, feature_names=feature_names)
    y_train = train_df["label"].astype(int)
    positive_rate = float(y_train.mean())
    positive_class_weight = None
    if class_imbalance:
        positive_class_weight = (1.0 - positive_rate) / max(positive_rate, 1e-9)
        sample_weight = np.where(y_train.to_numpy() == 1, positive_class_weight, 1.0)
        model.fit(train_df[feature_names], y_train, classifier__sample_weight=sample_weight)
    else:
        model.fit(train_df[feature_names], y_train)
    model_training_seconds = time.perf_counter() - fit_started_at

    if eval_balance:
        eval_subsets = build_balanced_eval_subsets(
            donations=donations,
            test_df=test_df,
            horizon_days=horizon_days,
            random_state=random_state,
        )
        prediction_frames: list[pd.DataFrame] = []
        per_user_frames: list[pd.DataFrame] = []
        overall_metrics = {
            "eval_balance": True,
            "eval_subset_count": 0,
        }
        aggregate_metrics: dict[str, list[float]] = {
            "roc_auc": [],
            "average_precision": [],
            "accuracy": [],
            "balanced_accuracy": [],
            "brier_score": [],
            "log_loss": [],
        }
        for subset_name, subset_df in eval_subsets.items():
            overall_metrics[f"{subset_name}_n_samples"] = int(len(subset_df))
            if subset_df.empty:
                for metric_name in aggregate_metrics:
                    overall_metrics[f"{subset_name}_{metric_name}"] = None
                continue
            subset_score = model.predict_proba(subset_df[feature_names])[:, 1]
            subset_pred = (subset_score >= 0.5).astype(int)
            subset_predictions = subset_df[["email", "anchor_date", "label", "future_donation_count"]].copy()
            subset_predictions["eval_subset"] = subset_name
            subset_predictions["score"] = subset_score
            subset_predictions["predicted_label"] = subset_pred
            prediction_frames.append(subset_predictions)

            subset_metrics = evaluate_predictions(subset_df["label"].astype(int), subset_score, subset_pred)
            for metric_name, metric_value in subset_metrics.items():
                overall_metrics[f"{subset_name}_{metric_name}"] = metric_value
                if metric_value is not None:
                    aggregate_metrics[metric_name].append(float(metric_value))

            subset_per_user = evaluate_by_user(subset_predictions)
            subset_per_user["eval_subset"] = subset_name
            per_user_frames.append(subset_per_user)
            overall_metrics["eval_subset_count"] = int(overall_metrics["eval_subset_count"]) + 1

        print_eval_balance_concat_debug(model_name, per_user_frames)
        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        cleaned_per_user_frames = [drop_all_na_columns(frame) for frame in per_user_frames]
        per_user = pd.concat(cleaned_per_user_frames, ignore_index=True) if cleaned_per_user_frames else pd.DataFrame()
        for metric_name, metric_values in aggregate_metrics.items():
            overall_metrics[f"eval_balance_mean_{metric_name}"] = (
                float(np.mean(metric_values)) if metric_values else None
            )
        mean_user_accuracy = overall_metrics["eval_balance_mean_accuracy"]
        mean_user_brier = overall_metrics["eval_balance_mean_brier_score"]
    else:
        test_score = model.predict_proba(test_df[feature_names])[:, 1]
        test_pred = (test_score >= 0.5).astype(int)

        predictions = test_df[["email", "anchor_date", "label"]].copy()
        predictions["score"] = test_score
        predictions["predicted_label"] = test_pred
        per_user = evaluate_by_user(predictions)
        overall_metrics = evaluate_predictions(test_df["label"].astype(int), test_score, test_pred)
        mean_user_accuracy = float(per_user["accuracy"].mean())
        mean_user_brier = float(per_user["brier_score"].mean())
    overall_metrics.update(
        {
            "model": model_name,
            "horizon_days": horizon_days,
            "slice_days": slice_days,
            "lookback_slices": lookback_slices,
            "anchor_stride_days": anchor_stride_days or slice_days,
            "n_total_samples": int(len(dataset)),
            "n_train_samples": int(len(train_df)),
            "n_test_samples": int(len(test_df)),
            "n_total_users": int(dataset["email"].nunique()),
            "n_train_users": int(train_df["email"].nunique()),
            "n_test_users": int(test_df["email"].nunique()),
            "training_positive_rate": float(train_df["label"].mean()),
            "test_positive_rate": float(test_df["label"].mean()),
            "mean_user_accuracy": mean_user_accuracy,
            "mean_user_brier": mean_user_brier,
            "dataset_processing_time_seconds": prepared.dataset_processing_seconds,
            "model_training_time_seconds": model_training_seconds,
            "total_pipeline_time_seconds": prepared.dataset_processing_seconds + model_training_seconds,
            "class_imbalance": class_imbalance,
            "positive_class_weight": positive_class_weight,
            "eval_balance": eval_balance,
            "normalize": apply_normalization,
            "normalization_method": normalization_method,
        }
    )

    model_dir = Path(output_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.joblib"
    joblib.dump(model, model_path)
    dataset.to_csv(model_dir / "sequence_dataset.csv", index=False)
    train_df.to_csv(model_dir / "train_sequence_dataset.csv", index=False)
    test_df.to_csv(model_dir / "test_sequence_dataset.csv", index=False)
    predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    per_user.to_csv(model_dir / "per_user_metrics.csv", index=False)
    save_json(model_dir / "summary_metrics.json", overall_metrics)
    save_json(
        model_dir / "model_metadata.json",
        {
            "model_name": model_name,
            "model_path": str(model_path),
            "horizon_days": horizon_days,
            "slice_days": slice_days,
            "lookback_slices": lookback_slices,
            "anchor_stride_days": anchor_stride_days or slice_days,
            "feature_names": feature_names,
            "normalize": apply_normalization,
            "normalization_method": normalization_method,
        },
    )

    return SequenceExperimentResult(
        overall_metrics=overall_metrics,
        per_user_metrics=per_user,
        predictions=predictions,
        dataset=dataset,
        model_dir=model_dir,
    )


def run_torch_sequence_experiment(
    model_name: str,
    model_builder: Callable[[int], nn.Module],
    xlsx_path: str | Path,
    output_root: str | Path = "outputs",
    horizon_days: int = 90,
    test_size: float = 0.25,
    slice_days: int = 30,
    lookback_slices: int = 6,
    min_examples_per_user: int = 2,
    anchor_stride_days: int | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    extra_metadata: dict | None = None,
    class_imbalance: bool = False,
    eval_balance: bool = False,
    normalize: bool = False,
) -> SequenceExperimentResult:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    prepared = load_or_prepare_sequence_data(
        xlsx_path=xlsx_path,
        output_root=output_root,
        horizon_days=horizon_days,
        test_size=test_size,
        slice_days=slice_days,
        lookback_slices=lookback_slices,
        min_examples_per_user=min_examples_per_user,
        anchor_stride_days=anchor_stride_days,
        random_state=random_state,
    )
    dataset = prepared.dataset
    train_df = prepared.train_df
    test_df = prepared.test_df
    donations = load_donations(xlsx_path)
    if train_df.empty or test_df.empty:
        raise ValueError("No train/test sequence samples were created. Adjust slice_days/lookback_slices/horizon_days.")
    print_training_debug(
        model_name,
        donations,
        train_df,
        test_df,
        horizon_days,
        slice_days,
        lookback_slices,
        anchor_stride_days or slice_days,
    )

    x_train = reshape_flat_features_to_sequence(train_df, lookback_slices)
    x_test = reshape_flat_features_to_sequence(test_df, lookback_slices)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    apply_normalization, normalization_method = resolve_torch_normalization(normalize_requested=normalize)
    train_mean = x_train.mean(axis=0, keepdims=True)
    train_std = x_train.std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    if apply_normalization:
        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    fit_started_at = time.perf_counter()
    model = model_builder(len(SLICE_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    positive_class_weight = (negatives / max(positives, 1.0)) if class_imbalance else None
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_class_weight, dtype=torch.float32))
        if class_imbalance
        else nn.BCEWithLogitsLoss()
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    model_training_seconds = time.perf_counter() - fit_started_at
    if eval_balance:
        eval_subsets = build_balanced_eval_subsets(
            donations=donations,
            test_df=test_df,
            horizon_days=horizon_days,
            random_state=random_state,
        )
        prediction_frames: list[pd.DataFrame] = []
        per_user_frames: list[pd.DataFrame] = []
        overall_metrics = {
            "eval_balance": True,
            "eval_subset_count": 0,
        }
        aggregate_metrics: dict[str, list[float]] = {
            "roc_auc": [],
            "average_precision": [],
            "accuracy": [],
            "balanced_accuracy": [],
            "brier_score": [],
            "log_loss": [],
        }
        with torch.no_grad():
            for subset_name, subset_df in eval_subsets.items():
                overall_metrics[f"{subset_name}_n_samples"] = int(len(subset_df))
                if subset_df.empty:
                    for metric_name in aggregate_metrics:
                        overall_metrics[f"{subset_name}_{metric_name}"] = None
                    continue

                subset_x = reshape_flat_features_to_sequence(subset_df, lookback_slices)
                if apply_normalization:
                    subset_x = (subset_x - train_mean) / train_std
                subset_logits = model(torch.tensor(subset_x)).squeeze(-1)
                subset_score = torch.sigmoid(subset_logits).cpu().numpy()
                subset_pred = (subset_score >= 0.5).astype(int)

                subset_predictions = subset_df[["email", "anchor_date", "label", "future_donation_count"]].copy()
                subset_predictions["eval_subset"] = subset_name
                subset_predictions["score"] = subset_score
                subset_predictions["predicted_label"] = subset_pred
                prediction_frames.append(subset_predictions)

                subset_metrics = evaluate_predictions(subset_df["label"].astype(int), subset_score, subset_pred)
                for metric_name, metric_value in subset_metrics.items():
                    overall_metrics[f"{subset_name}_{metric_name}"] = metric_value
                    if metric_value is not None:
                        aggregate_metrics[metric_name].append(float(metric_value))

                subset_per_user = evaluate_by_user(subset_predictions)
                subset_per_user["eval_subset"] = subset_name
                per_user_frames.append(subset_per_user)
                overall_metrics["eval_subset_count"] = int(overall_metrics["eval_subset_count"]) + 1

        print_eval_balance_concat_debug(model_name, per_user_frames)
        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        cleaned_per_user_frames = [drop_all_na_columns(frame) for frame in per_user_frames]
        per_user = pd.concat(cleaned_per_user_frames, ignore_index=True) if cleaned_per_user_frames else pd.DataFrame()
        for metric_name, metric_values in aggregate_metrics.items():
            overall_metrics[f"eval_balance_mean_{metric_name}"] = (
                float(np.mean(metric_values)) if metric_values else None
            )
        mean_user_accuracy = overall_metrics["eval_balance_mean_accuracy"]
        mean_user_brier = overall_metrics["eval_balance_mean_brier_score"]
    else:
        with torch.no_grad():
            test_logits = model(torch.tensor(x_test)).squeeze(-1)
            test_score = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_score >= 0.5).astype(int)

        predictions = test_df[["email", "anchor_date", "label"]].copy()
        predictions["score"] = test_score
        predictions["predicted_label"] = test_pred
        per_user = evaluate_by_user(predictions)
        overall_metrics = evaluate_predictions(test_df["label"].astype(int), test_score, test_pred)
        mean_user_accuracy = float(per_user["accuracy"].mean())
        mean_user_brier = float(per_user["brier_score"].mean())
    overall_metrics.update(
        {
            "model": model_name,
            "horizon_days": horizon_days,
            "slice_days": slice_days,
            "lookback_slices": lookback_slices,
            "anchor_stride_days": anchor_stride_days or slice_days,
            "n_total_samples": int(len(dataset)),
            "n_train_samples": int(len(train_df)),
            "n_test_samples": int(len(test_df)),
            "n_total_users": int(dataset["email"].nunique()),
            "n_train_users": int(train_df["email"].nunique()),
            "n_test_users": int(test_df["email"].nunique()),
            "training_positive_rate": float(train_df["label"].mean()),
            "test_positive_rate": float(test_df["label"].mean()),
            "mean_user_accuracy": mean_user_accuracy,
            "mean_user_brier": mean_user_brier,
            "dataset_processing_time_seconds": prepared.dataset_processing_seconds,
            "model_training_time_seconds": model_training_seconds,
            "total_pipeline_time_seconds": prepared.dataset_processing_seconds + model_training_seconds,
            "class_imbalance": class_imbalance,
            "positive_class_weight": positive_class_weight,
            "eval_balance": eval_balance,
            "normalize": apply_normalization,
            "normalization_method": normalization_method,
        }
    )

    model_dir = Path(output_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_torch_checkpoint(
        model_dir=model_dir,
        model_name=model_name,
        model=model,
        horizon_days=horizon_days,
        slice_days=slice_days,
        lookback_slices=lookback_slices,
        train_mean=train_mean.squeeze(0),
        train_std=train_std.squeeze(0),
        normalize=apply_normalization,
        normalization_method=normalization_method,
        extra_metadata=extra_metadata,
    )
    dataset.to_csv(model_dir / "sequence_dataset.csv", index=False)
    train_df.to_csv(model_dir / "train_sequence_dataset.csv", index=False)
    test_df.to_csv(model_dir / "test_sequence_dataset.csv", index=False)
    predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    per_user.to_csv(model_dir / "per_user_metrics.csv", index=False)
    save_json(model_dir / "summary_metrics.json", overall_metrics)
    save_json(
        model_dir / "model_metadata.json",
        {
            "model_name": model_name,
            "model_path": str(model_path),
            "horizon_days": horizon_days,
            "slice_days": slice_days,
            "lookback_slices": lookback_slices,
            "anchor_stride_days": anchor_stride_days or slice_days,
            "feature_names": make_sequence_feature_names(lookback_slices),
            "backend": "torch",
            "extra_metadata": extra_metadata or {},
            "normalize": apply_normalization,
            "normalization_method": normalization_method,
        },
    )
    return SequenceExperimentResult(overall_metrics, per_user, predictions, dataset, model_dir)


def score_user_with_saved_sequence_model(
    model_name: str,
    xlsx_path: str | Path,
    output_root: str | Path,
    email: str,
    horizon_days: int | None = None,
    normalize: bool = False,
) -> dict:
    normalized_email = email.strip().lower()
    model_dir = Path(output_root) / model_name
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Saved model metadata not found at {metadata_path}. "
            f"This usually means `{model_name}` has not been trained yet for output_root=`{output_root}`. "
            f"Run `python scripts/train_{model_name}.py --xlsx-path donation_list.xlsx --output-root {output_root} ...` "
            f"or rerun `python scripts/run_all_models.py --output-root {output_root} ...` after the latest code changes."
        )

    metadata = json.loads(metadata_path.read_text())
    saved_normalize, normalization_method = infer_saved_tabular_normalization(metadata)
    if normalize != saved_normalize:
        raise ValueError(
            f"Normalization flag mismatch for saved model `{model_name}` under `{output_root}`. "
            f"Saved model normalize={saved_normalize}, but test request normalize={normalize}. "
            f"Retrain and test with matching `--normalize` usage."
        )
    saved_horizon = int(metadata["horizon_days"])
    if horizon_days is not None and horizon_days != saved_horizon:
        raise ValueError(
            f"Requested horizon_days={horizon_days} does not match saved model horizon_days={saved_horizon}"
        )

    donations = load_donations(xlsx_path)
    user_history = donations.loc[donations["email"] == normalized_email].copy()
    if user_history.empty:
        raise ValueError(f"No donation history found for {normalized_email}")
    user_history = user_history.sort_values("donation_date").reset_index(drop=True)

    anchor_date = user_history["donation_date"].max()
    feature_row, slice_debug = build_sequence_example(
        user_history=user_history,
        anchor_date=anchor_date,
        slice_days=int(metadata["slice_days"]),
        lookback_slices=int(metadata["lookback_slices"]),
    )
    feature_frame = pd.DataFrame([feature_row], columns=metadata["feature_names"])
    model_path = Path(metadata["model_path"])
    if not model_path.exists():
        candidate = model_dir / model_path.name
        if candidate.exists():
            model_path = candidate
        else:
            raise FileNotFoundError(
                f"Saved model file not found at {metadata['model_path']} or {candidate}. "
                f"The metadata likely points to an older output-root. Retrain `{model_name}` into `{output_root}` if needed."
            )
    model = joblib.load(model_path)
    probability = float(model.predict_proba(feature_frame)[0, 1])

    return {
        "model_name": model_name,
        "email": normalized_email,
        "horizon_days": saved_horizon,
        "slice_days": int(metadata["slice_days"]),
        "lookback_slices": int(metadata["lookback_slices"]),
        "anchor_stride_days": int(metadata.get("anchor_stride_days", metadata["slice_days"])),
        "normalize": saved_normalize,
        "normalization_method": normalization_method,
        "model_path": str(model_path),
        "history": user_history,
        "history_stats": summarize_user_history_stats(user_history),
        "slice_debug": slice_debug,
        "features": feature_frame,
        "probability": probability,
    }


def score_user_with_saved_torch_model(
    model_name: str,
    model_builder: Callable[[dict], nn.Module],
    xlsx_path: str | Path,
    output_root: str | Path,
    email: str,
    horizon_days: int | None = None,
    normalize: bool = False,
) -> dict:
    normalized_email = email.strip().lower()
    model_dir = Path(output_root) / model_name
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Saved model metadata not found at {metadata_path}. "
            f"This usually means `{model_name}` has not been trained yet for output_root=`{output_root}`. "
            f"Run `python scripts/train_{model_name}.py --xlsx-path donation_list.xlsx --output-root {output_root} ...` "
            f"or rerun `python scripts/run_all_models.py --output-root {output_root} ...` after the latest code changes."
        )
    metadata = json.loads(metadata_path.read_text())
    saved_horizon = int(metadata["horizon_days"])
    if horizon_days is not None and horizon_days != saved_horizon:
        raise ValueError(
            f"Requested horizon_days={horizon_days} does not match saved model horizon_days={saved_horizon}"
        )

    donations = load_donations(xlsx_path)
    user_history = donations.loc[donations["email"] == normalized_email].copy()
    if user_history.empty:
        raise ValueError(f"No donation history found for {normalized_email}")
    user_history = user_history.sort_values("donation_date").reset_index(drop=True)
    anchor_date = user_history["donation_date"].max()
    feature_row, slice_debug = build_sequence_example(
        user_history=user_history,
        anchor_date=anchor_date,
        slice_days=int(metadata["slice_days"]),
        lookback_slices=int(metadata["lookback_slices"]),
    )
    feature_frame = pd.DataFrame([feature_row], columns=metadata["feature_names"])
    sequence = reshape_flat_features_to_sequence(feature_frame, int(metadata["lookback_slices"]))

    model_path = Path(metadata["model_path"])
    if not model_path.exists():
        candidate = model_dir / model_path.name
        if candidate.exists():
            model_path = candidate
        else:
            raise FileNotFoundError(
                f"Saved model file not found at {metadata['model_path']} or {candidate}. "
                f"The metadata likely points to an older output-root. Retrain `{model_name}` into `{output_root}` if needed."
            )
    checkpoint = torch.load(model_path, map_location="cpu")
    saved_normalize, normalization_method = infer_saved_torch_normalization(metadata, checkpoint)
    if normalize != saved_normalize:
        raise ValueError(
            f"Normalization flag mismatch for saved model `{model_name}` under `{output_root}`. "
            f"Saved model normalize={saved_normalize}, but test request normalize={normalize}. "
            f"Retrain and test with matching `--normalize` usage."
        )
    mean = np.array(checkpoint["train_mean"], dtype=np.float32).reshape(1, int(metadata["lookback_slices"]), len(SLICE_FEATURES))
    std = np.array(checkpoint["train_std"], dtype=np.float32).reshape(1, int(metadata["lookback_slices"]), len(SLICE_FEATURES))
    if saved_normalize:
        sequence = (sequence - mean) / std
    model = model_builder(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probability = float(torch.sigmoid(model(torch.tensor(sequence))).squeeze().item())

    return {
        "model_name": model_name,
        "email": normalized_email,
        "horizon_days": saved_horizon,
        "slice_days": int(metadata["slice_days"]),
        "lookback_slices": int(metadata["lookback_slices"]),
        "anchor_stride_days": int(metadata.get("anchor_stride_days", metadata["slice_days"])),
        "normalize": saved_normalize,
        "normalization_method": normalization_method,
        "model_path": str(model_path),
        "history": user_history,
        "history_stats": summarize_user_history_stats(user_history),
        "slice_debug": slice_debug,
        "features": feature_frame,
        "probability": probability,
    }


def score_user_with_saved_onnx_model(
    model_name: str,
    xlsx_path: str | Path,
    output_root: str | Path,
    email: str,
    horizon_days: int | None = None,
    normalize: bool = False,
    export_subdir: str = "exported",
    exported_model: str | None = None,
) -> dict:
    import onnxruntime as ort

    normalized_email = email.strip().lower()
    model_dir = Path(output_root) / model_name
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Saved model metadata not found at {metadata_path}. "
            f"This usually means `{model_name}` has not been trained yet for output_root=`{output_root}`. "
            f"Run `python scripts/train_{model_name}.py --xlsx-path donation_list.xlsx --output-root {output_root} ...` "
            f"or rerun `python scripts/run_all_models.py --output-root {output_root} ...` after the latest code changes."
        )
    metadata = json.loads(metadata_path.read_text())
    saved_horizon = int(metadata["horizon_days"])
    if horizon_days is not None and horizon_days != saved_horizon:
        raise ValueError(
            f"Requested horizon_days={horizon_days} does not match saved model horizon_days={saved_horizon}"
        )

    export_dir = model_dir / export_subdir
    export_metadata_path = export_dir / "portable_metadata.json"
    export_metadata = json.loads(export_metadata_path.read_text()) if export_metadata_path.exists() else {}

    if exported_model is not None:
        onnx_path = Path(exported_model)
    elif export_metadata.get("exported_model_path"):
        onnx_path = Path(str(export_metadata["exported_model_path"]))
    else:
        candidates = [
            export_dir / "transformer_export_int8.onnx",
            export_dir / "transformer_export.onnx",
            export_dir / "transformer_pruned_int8.onnx",
            export_dir / "transformer_pruned.onnx",
        ]
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is None:
            raise FileNotFoundError(
                f"No exported ONNX model found under {export_dir}. "
                f"Run `python scripts/export_transformer_portable.py --output-root {output_root} ...` first."
            )
        onnx_path = existing
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Exported ONNX model not found at {onnx_path}. "
            f"Run `python scripts/export_transformer_portable.py --output-root {output_root} ...` first."
        )

    donations = load_donations(xlsx_path)
    user_history = donations.loc[donations["email"] == normalized_email].copy()
    if user_history.empty:
        raise ValueError(f"No donation history found for {normalized_email}")
    user_history = user_history.sort_values("donation_date").reset_index(drop=True)
    anchor_date = user_history["donation_date"].max()
    feature_row, slice_debug = build_sequence_example(
        user_history=user_history,
        anchor_date=anchor_date,
        slice_days=int(metadata["slice_days"]),
        lookback_slices=int(metadata["lookback_slices"]),
    )
    feature_frame = pd.DataFrame([feature_row], columns=metadata["feature_names"])
    sequence = reshape_flat_features_to_sequence(feature_frame, int(metadata["lookback_slices"]))

    checkpoint_path = Path(metadata["model_path"])
    if not checkpoint_path.exists():
        candidate = model_dir / checkpoint_path.name
        if candidate.exists():
            checkpoint_path = candidate
        else:
            raise FileNotFoundError(
                f"Saved model file not found at {metadata['model_path']} or {candidate}. "
                f"The metadata likely points to an older output-root. Retrain `{model_name}` into `{output_root}` if needed."
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_normalize, normalization_method = infer_saved_torch_normalization(metadata, checkpoint)
    if normalize != saved_normalize:
        raise ValueError(
            f"Normalization flag mismatch for saved ONNX export `{model_name}` under `{output_root}`. "
            f"Saved model normalize={saved_normalize}, but test request normalize={normalize}. "
            f"Retrain/export and test with matching `--normalize` usage."
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    probability = float(session.run(None, {input_name: sequence.astype(np.float32)})[0].reshape(-1)[0])

    return {
        "model_name": f"{model_name}_onnx",
        "email": normalized_email,
        "horizon_days": saved_horizon,
        "slice_days": int(metadata["slice_days"]),
        "lookback_slices": int(metadata["lookback_slices"]),
        "anchor_stride_days": int(metadata.get("anchor_stride_days", metadata["slice_days"])),
        "normalize": saved_normalize,
        "normalization_method": normalization_method,
        "model_path": str(onnx_path),
        "history": user_history,
        "history_stats": summarize_user_history_stats(user_history),
        "slice_debug": slice_debug,
        "features": feature_frame,
        "probability": probability,
    }


def _dataframe_to_rich_table(title: str, frame: pd.DataFrame, max_rows: int | None = None) -> Table:
    table = Table(title=title, show_lines=False)
    for column in frame.columns:
        table.add_column(str(column))

    render_frame = frame.head(max_rows) if max_rows is not None else frame
    for _, row in render_frame.iterrows():
        table.add_row(*[str(value) for value in row.tolist()])

    return table


def _format_debug_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[column]):
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def _numeric_histogram_table(title: str, values: pd.Series, bins: int = 8, bar_width: int = 24) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("range")
    table.add_column("count")
    table.add_column("hist")
    bar_glyph = "█"
    bar_style = "bright_cyan"

    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        table.add_row("n/a", "0", "")
        return table

    if len(numeric) == 1 or float(numeric.min()) == float(numeric.max()):
        label = f"{float(numeric.iloc[0]):.3f}"
        bar = f"[{bar_style}]{bar_glyph * min(bar_width, max(1, len(numeric)))}[/]"
        table.add_row(label, str(int(len(numeric))), bar)
        return table

    counts, edges = np.histogram(numeric.to_numpy(), bins=min(bins, len(numeric)))
    max_count = int(counts.max()) if len(counts) else 0
    for idx, count in enumerate(counts):
        left = edges[idx]
        right = edges[idx + 1]
        bar_len = int(round((int(count) / max_count) * bar_width)) if max_count > 0 else 0
        bar = f"[{bar_style}]{bar_glyph * max(1, bar_len)}[/]" if count > 0 else ""
        table.add_row(f"[{left:.3f}, {right:.3f})", str(int(count)), bar)
    return table


def _render_donation_trend(history_df: pd.DataFrame, width: int = 40, max_gap: int = 2) -> None:
    """Generates a compressed blue bar trend using dots to represent long gaps."""
    console = Console()
    df = history_df.copy()
    df['donation_date'] = pd.to_datetime(df['donation_date'])
    
    # Aggregate donations by date
    trend = df.groupby('donation_date')['amount'].sum().sort_index()
    if trend.empty:
        return

    max_val = trend.max()
    chart_rows = []
    dates = trend.index.tolist()

    for i in range(len(dates)):
        current_date = dates[i]
        value = trend[current_date]
        
        # 1. Add the actual donation bar
        bar_length = int((value / max_val) * width) if max_val > 0 else 0
        bar_char = "▓" * bar_length
        date_str = current_date.strftime("%Y-%m-%d")
        chart_rows.append(f"{date_str} | [bold blue]{bar_char:<{width}}[/bold blue] ${value:>8.2f}")

        # 2. Handle the gap to the next donation
        if i < len(dates) - 1:
            next_date = dates[i+1]
            gap_days = (next_date - current_date).days - 1
            
            if gap_days > 0:
                if gap_days <= max_gap:
                    # Small gap: show the empty days normally
                    for day_offset in range(1, gap_days + 1):
                        empty_date = (current_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
                        chart_rows.append(f"{empty_date} | [dim]{' ' * width}[/dim] $    0.00")
                else:
                    # Large gap: compress into a "dots" row
                    chart_rows.append(f"           [dim]⋮ ({gap_days} days gap)[/dim]")

    console.print(
        Panel(
            "\n".join(chart_rows),
            title="[bold cyan]Donation Trend (Compressed)[/bold cyan]",
            border_style="blue",
            expand=False
        )
    )


def print_score_debug(result: dict) -> None:
    console = Console()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Model:[/bold] {result['model_name']}",
                    f"[bold]Email:[/bold] {result['email']}",
                    f"[bold]Horizon Days:[/bold] {result['horizon_days']}",
                    f"[bold]Slice Days:[/bold] {result['slice_days']}",
                    f"[bold]Lookback Slices:[/bold] {result['lookback_slices']}",
                    f"[bold]Anchor Stride Days:[/bold] {result.get('anchor_stride_days', result['slice_days'])}",
                    f"[bold]Normalized Features:[/bold] {result.get('normalize', False)} ({result.get('normalization_method', 'none')})",
                    f"[bold]Model Path:[/bold] {result['model_path']}",
                ]
            ),
            title="Debug Summary",
            border_style="cyan",
        )
    )
    console.print(
        _dataframe_to_rich_table(
            "User History",
            _format_debug_frame(result["history"][["donation_date", "amount"]]),
        )
    )

    # Add the trend visualization here
    _render_donation_trend(result["history"])


    console.print(_dataframe_to_rich_table("User History Statistics", _format_debug_frame(result["history_stats"])))
    gap_days = (
        pd.to_datetime(result["history"]["donation_date"], errors="coerce").sort_values().diff().dt.days.dropna()
    )
    console.print(_numeric_histogram_table("Donation Amount Histogram", result["history"]["amount"]))
    console.print(_numeric_histogram_table("Donation Gap Days Histogram", gap_days))

    console.print(
        Panel.fit(
            f"[bold green]Probability of donation within {result['horizon_days']} days:[/bold green] {result['probability']:.6f}",
            border_style="green",
        )
    )
