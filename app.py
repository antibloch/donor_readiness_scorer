from __future__ import annotations

import tempfile
from pathlib import Path

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


def ensure_required_files(
    output_root: str,
    model_name: str,
    export_subdir: str,
    exported_model: str | None,
) -> tuple[bool, list[str], str | None]:
    errors: list[str] = []

    model_dir = Path(output_root) / model_name
    metadata_path = model_dir / "model_metadata.json"
    export_dir = model_dir / export_subdir

    if not metadata_path.exists():
        errors.append(f"Missing model metadata: `{metadata_path}`")

    if exported_model:
        onnx_path = Path(exported_model)
        if not onnx_path.exists():
            errors.append(f"Specified exported model not found: `{onnx_path}`")
        resolved_model = str(onnx_path)
    else:
        if not export_dir.exists():
            errors.append(f"Missing export directory: `{export_dir}`")

        candidates = [
            export_dir / "transformer_export_int8.onnx",
            export_dir / "transformer_export.onnx",
            export_dir / "transformer_pruned_int8.onnx",
            export_dir / "transformer_pruned.onnx",
        ]
        existing = next((p for p in candidates if p.exists()), None)
        resolved_model = str(existing) if existing else None

        if existing is None:
            errors.append(
                "No ONNX model found. Expected one of: "
                + ", ".join(f"`{p}`" for p in candidates)
            )

    return len(errors) == 0, errors, resolved_model


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
            f"Invalid `donation_date` in row(s): {bad_rows}. Use YYYY-MM-DD format."
        )

    if invalid_amount_rows.any():
        bad_rows = (cleaned.index[invalid_amount_rows] + 1).tolist()
        errors.append(f"Invalid `amount` in row(s): {bad_rows}. Use numeric values.")

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


def render_result(result: dict, cleaned_df: pd.DataFrame, resolved_model: str | None) -> None:
    st.success("Scoring completed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability", f"{result['probability']:.6f}")
    c2.metric("Horizon days", str(result["horizon_days"]))
    c3.metric("Model", result["model_name"])

    st.subheader("Input donation history")
    st.dataframe(cleaned_df, use_container_width=True)

    if not cleaned_df.empty:
        chart_df = cleaned_df.copy()
        chart_df["donation_date"] = pd.to_datetime(chart_df["donation_date"])
        chart_df = chart_df.sort_values("donation_date").set_index("donation_date")[["amount"]]
        st.subheader("Donation amounts over time")
        st.line_chart(chart_df)

    st.subheader("Inference summary")
    st.json(
        {
            "model_name": result["model_name"],
            "model_path": resolved_model or result.get("model_path"),
            "horizon_days": result["horizon_days"],
            "slice_days": result["slice_days"],
            "lookback_slices": result["lookback_slices"],
            "anchor_stride_days": result["anchor_stride_days"],
            "normalize": result["normalize"],
            "normalization_method": result["normalization_method"],
            "probability": result["probability"],
        }
    )

    st.subheader("History statistics")
    st.dataframe(result["history_stats"], use_container_width=True)

    st.subheader("Slice debug")
    st.dataframe(result["slice_debug"], use_container_width=True)

    st.subheader("Feature vector")
    st.dataframe(result["features"], use_container_width=True)


def main() -> None:
    st.title("Donor Readiness Scorer")
    st.write(
        "Enter a donor's donation history as rows of `donation_date` and `amount`, then run ONNX inference."
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
        normalize = st.checkbox("Normalize", value=False)
        horizon_days = st.number_input(
            "Horizon days",
            min_value=1,
            value=90,
            step=1,
        )

    ok, file_errors, resolved_model = ensure_required_files(
        output_root=output_root,
        model_name=model_name,
        export_subdir=export_subdir,
        exported_model=exported_model,
    )

    if not ok:
        st.error("Deployment files are incomplete or misconfigured.")
        for err in file_errors:
            st.write(f"- {err}")
        st.stop()

    st.subheader("Donation history input")

    starter_df = pd.DataFrame(
        [
            {"donation_date": "2025-01-15", "amount": 100.0},
            {"donation_date": "2025-03-01", "amount": 50.0},
            {"donation_date": "2025-06-20", "amount": 150.0},
        ]
    )

    edited_df = st.data_editor(
        starter_df,
        num_rows="dynamic",
        use_container_width=True,
        key="donation_editor",
    )

    st.caption("Use columns `donation_date` and `amount`. Dates must be in YYYY-MM-DD format.")

    if st.button("Run score", type="primary"):
        cleaned_df, validation_errors = validate_user_table(edited_df)

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()

        temp_xlsx_path = build_scoring_xlsx(cleaned_df)

        try:
            result = score_user_with_saved_onnx_model(
                model_name=model_name,
                xlsx_path=temp_xlsx_path,
                output_root=output_root,
                email=INTERNAL_EMAIL,
                horizon_days=int(horizon_days),
                normalize=normalize,
                export_subdir=export_subdir,
                exported_model=resolved_model,
            )
            render_result(result, cleaned_df, resolved_model)

        except Exception as e:
            st.exception(e)

        finally:
            try:
                Path(temp_xlsx_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
