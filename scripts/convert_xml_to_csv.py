from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    from pune_aqi.io.xml_to_csv import xml_to_csv  # noqa: E402

    parser = argparse.ArgumentParser(description="Convert CPCB AQI XML to CSV.")
    parser.add_argument("--xml", required=True, help="Path to input XML file")
    parser.add_argument(
        "--out",
        default=str(repo_root / "data" / "processed" / "aqi_from_xml.csv"),
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    res = xml_to_csv(Path(args.xml), output_csv=Path(args.out))
    print(f"Wrote {res.rows} rows to {res.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

