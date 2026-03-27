from __future__ import annotations

import argparse

from common import print_score_debug, score_user_with_saved_onnx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a saved ONNX transformer model and score one user.")
    parser.add_argument("--xlsx-path", default="donation_list.xlsx")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--email", required=True)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--export-subdir", default="exported")
    parser.add_argument("--exported-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_user_with_saved_onnx_model(
        model_name="transformer",
        xlsx_path=args.xlsx_path,
        output_root=args.output_root,
        email=args.email,
        horizon_days=args.horizon_days,
        normalize=args.normalize,
        export_subdir=args.export_subdir,
        exported_model=args.exported_model,
    )
    print_score_debug(result)


if __name__ == "__main__":
    main()
