from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from common import score_user_with_saved_onnx_model


st.set_page_config(
    page_title="Donor Readiness Scorer",
    page_icon="📈",
    layout="wide",
)

DEFAULT_OUTPUT_ROOT = "outputs_sequence"
DEFAULT_MODEL_NAME = "transformer"
DEFAULT_EXPORT_SUBDIR = "exported"
INTERNAL_EMAIL = "demo_user@example.com"


def load_model_metadata(output_root: str, model_name: str) -> dict:
    model_dir = Path(output_root) / model_name
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def resolve_onnx_path(
    output_root: str,
    model_name: str,
    export_subdir: str,
    exported_model: str | None,
) -> str:
    model_dir = Path(output_root) / model_name
    export_dir = model_dir / export_subdir

    if exported_model:
        onnx_path = Path(exported_model)
        if not onnx_path.exists():
            raise FileNotFoundError(f"Specified exported model not found: {onnx_path}")
        return str(onnx_path)

    export_metadata_path = export_dir / "portable_metadata.json"
    if export_metadata_path.exists():
        export_metadata = json.loads(export_metadata_path.read_text())
        exported_model_path = export_metadata.get("exported_model_path")
        if exported_model_path and Path(exported_model_path).exists():
            return str(Path(exported_model_path))

    candidates = [
        export_dir / "transformer_export_int8.onnx",
        export_dir / "transformer_export.onnx",
        export_dir / "transformer_pruned_int8.onnx",
        export_dir / "transformer_pruned.onnx",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is None:
        raise FileNotFoundError(
            f"No ONNX model found under {export_dir}. Expected one of: "
            + ", ".join(str(p.name) for p in candidates)
        )
    return str(existing)


def validate_user_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []

    required_columns = ["donation_date", "amount"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return df, [f"Missing required column(s): {', '.join(missing)}"]

    cleaned = df.copy()
    cleaned["donation_date"] = pd.to_datetime(
        cleaned["donation_date"], errors="coerce", format="%Y-%m-%d"
    )
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")

    invalid_date_rows = cleaned["donation_date"].isna()
    invalid_amount_rows = cleaned["amount"].isna()

    if invalid_date_rows.any():
        bad_rows = (cleaned.index[invalid_date_rows] + 1).tolist()
        errors.append(
            f"Invalid donation_date in row(s): {bad_rows}. Use YYYY-MM-DD format."
        )

    if invalid_amount_rows.any():
        bad_rows = (cleaned.index[invalid_amount_rows] + 1).tolist()
        errors.append(f"Invalid amount in row(s): {bad_rows}. Use numeric values.")

    cleaned = cleaned.dropna(subset=["donation_date", "amount"]).copy()

    if cleaned.empty:
        errors.append("Please enter at least one valid donation row.")
        return cleaned, errors

    cleaned["amount"] = cleaned["amount"].astype(float)
    cleaned = cleaned.sort_values("donation_date").reset_index(drop=True)
    return cleaned, errors


def build_scoring_xlsx(user_history: pd.DataFrame) -> str:
    scoring_df = user_history.copy()
    scoring_df["email"] = INTERNAL_EMAIL
    scoring_df = scoring_df[["email", "donation_date", "amount"]]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp_path = tmp.name
    tmp.close()

    scoring_df.to_excel(tmp_path, index=False)
    return tmp_path


def build_gap_histogram_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    gap_df = cleaned_df.copy()
    gap_df["donation_date"] = pd.to_datetime(gap_df["donation_date"])
    gap_df = gap_df.sort_values("donation_date").reset_index(drop=True)
    gap_df["gap_days"] = gap_df["donation_date"].diff().dt.days
    gap_df = gap_df.dropna(subset=["gap_days"]).copy()
    if gap_df.empty:
        return gap_df
    gap_df["gap_days"] = gap_df["gap_days"].astype(float)
    return gap_df


def make_histogram_series(
    values: np.ndarray,
    bins: int = 10,
    integer_labels: bool = True,
) -> pd.Series:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return pd.Series(dtype=int)

    if values.size == 1 or float(values.min()) == float(values.max()):
        v = values[0]
        if integer_labels:
            label = f"[{int(v)}-{int(v)}]"
        else:
            label = f"[{v:.2f}-{v:.2f}]"
        return pd.Series([int(values.size)], index=[label])

    hist, edges = np.histogram(values, bins=bins)

    labels = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        if integer_labels:
            left_label = int(np.floor(left))
            right_label = int(np.ceil(right))
            label = f"[{left_label}-{right_label}]"
        else:
            label = f"[{left:.2f}-{right:.2f}]"

        labels.append(label)

    return pd.Series(hist, index=labels)


def render_result(result: dict, cleaned_df: pd.DataFrame) -> None:
    st.success("Scoring completed.")

    horizon_days = int(result["horizon_days"])
    prob_pct = float(result["probability"]) * 100.0

    st.subheader(f"Probability of donation in next {horizon_days} days")
    st.metric(
        label=f"Probability of donation in next {horizon_days} days",
        value=f"{prob_pct:.2f}%",
    )

    st.subheader("Input donation history")
    display_df = cleaned_df.copy()
    display_df["donation_date"] = pd.to_datetime(display_df["donation_date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True)

    chart_df = cleaned_df.copy()
    chart_df["donation_date"] = pd.to_datetime(chart_df["donation_date"])
    chart_df = chart_df.sort_values("donation_date").set_index("donation_date")[["amount"]]

    st.subheader("Donation amounts over time")
    st.bar_chart(chart_df)

    st.subheader("Donation amount distribution")
    amount_hist_series = make_histogram_series(
        cleaned_df["amount"].to_numpy(),
        bins=min(10, max(1, len(cleaned_df))),
        integer_labels=False,
    )
    amount_hist_df = amount_hist_series.rename("count").to_frame()
    st.bar_chart(amount_hist_df)

    st.subheader("Donation date gap distribution")
    gap_df = build_gap_histogram_df(cleaned_df)
    if gap_df.empty:
        st.info("At least two donations are needed to compute donation date gaps.")
    else:
        gap_hist_series = make_histogram_series(
            gap_df["gap_days"].to_numpy(),
            bins=min(10, max(1, len(gap_df))),
            integer_labels=True,
        )
        gap_hist_df = gap_hist_series.rename("count").to_frame()
        st.bar_chart(gap_hist_df)


def main() -> None:
    st.title("Donor Readiness Scorer")
    st.write(
        "Enter donation history as rows of `donation_date` and `amount`, then run ONNX inference."
    )

    with st.sidebar:
        st.header("Model configuration")
        output_root = st.text_input("Output root", value=DEFAULT_OUTPUT_ROOT)
        model_name = st.text_input("Model name", value=DEFAULT_MODEL_NAME)
        export_subdir = st.text_input("Export subdir", value=DEFAULT_EXPORT_SUBDIR)
        exported_model = st.text_input(
            "Explicit ONNX model path (optional)",
            value="",
            help="Leave blank to auto-detect under outputs_sequence/transformer/exported/",
        ).strip() or None

    try:
        metadata = load_model_metadata(output_root, model_name)
        resolved_model = resolve_onnx_path(
            output_root=output_root,
            model_name=model_name,
            export_subdir=export_subdir,
            exported_model=exported_model,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    saved_horizon = int(metadata["horizon_days"])
    saved_normalize = bool(metadata.get("normalize", False))

    with st.sidebar:
        st.markdown("---")
        st.subheader("Saved model settings")
        st.write(f"Saved horizon_days: `{saved_horizon}`")
        st.write(f"Saved normalize: `{saved_normalize}`")

        horizon_days = st.number_input(
            "Horizon days",
            min_value=1,
            value=saved_horizon,
            step=1,
            help="Must match the saved model metadata.",
        )
        normalize = st.checkbox(
            "Normalize",
            value=saved_normalize,
            help="Must match the saved model metadata.",
        )

    st.subheader("Donation history input")

    starter_df = pd.DataFrame(
        [
            {"donation_date": "2025-01-15", "amount": 30.0},
            {"donation_date": "2025-03-01", "amount": 30.0},
            {"donation_date": "2025-06-20", "amount": 150.0},
        ]
    )

    edited_df = st.data_editor(
        starter_df,
        num_rows="dynamic",
        use_container_width=True,
        key="donation_editor",
    )

    st.caption("Use columns donation_date and amount. Dates must be YYYY-MM-DD.")

    if st.button("Run score", type="primary"):
        cleaned_df, validation_errors = validate_user_table(edited_df)

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()

        if horizon_days != saved_horizon:
            st.error(
                f"This model was trained for horizon_days={saved_horizon}. "
                f"Set the sidebar value to {saved_horizon}."
            )
            st.stop()

        if normalize != saved_normalize:
            st.error(
                f"This model was trained with normalize={saved_normalize}. "
                f"Set the sidebar checkbox to {saved_normalize}."
            )
            st.stop()

        temp_xlsx_path = build_scoring_xlsx(cleaned_df)

        try:
            result = score_user_with_saved_onnx_model(
                model_name=model_name,
                xlsx_path=temp_xlsx_path,
                output_root=output_root,
                email=INTERNAL_EMAIL,
                horizon_days=horizon_days,
                normalize=normalize,
                export_subdir=export_subdir,
                exported_model=resolved_model,
            )
            render_result(result, cleaned_df)
        except Exception as e:
            st.exception(e)
        finally:
            try:
                Path(temp_xlsx_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
