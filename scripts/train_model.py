from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    from pune_aqi.data.loaders import load_default_dataset  # noqa: E402
    from pune_aqi.models.train import results_to_frame, train_and_evaluate_models  # noqa: E402

    parser = argparse.ArgumentParser(description="Train and compare AQI regressors.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    df = load_default_dataset(repo_root=repo_root)
    results = train_and_evaluate_models(df, test_size=args.test_size, random_state=args.random_state)
    table = results_to_frame(results, split="test")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

