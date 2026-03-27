from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

def reshape_flat_features_to_sequence(frame: pd.DataFrame, lookback_slices: int) -> np.ndarray:
    feature_names = make_sequence_feature_names(lookback_slices)
    array = frame[feature_names].to_numpy(dtype=np.float32)
    return array.reshape(len(frame), lookback_slices, len(SLICE_FEATURES))

def infer_saved_torch_normalization(metadata: dict, checkpoint: dict) -> tuple[bool, str]:
    if "normalize" in metadata:
        apply_normalization = bool(metadata["normalize"])
        return apply_normalization, str(metadata.get("normalization_method", "trainset_zscore" if apply_normalization else "none"))
    if "normalize" in checkpoint:
        apply_normalization = bool(checkpoint["normalize"])
        return apply_normalization, str(checkpoint.get("normalization_method", "trainset_zscore" if apply_normalization else "none"))
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
