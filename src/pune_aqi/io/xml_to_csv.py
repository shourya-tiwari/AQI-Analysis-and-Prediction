from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class XmlToCsvResult:
    rows: int
    output_csv: Path | None


def parse_cpcb_xml(xml_path: Path) -> pd.DataFrame:
    """Parse CPCB-like AQI XML into a tabular DataFrame.

    Expected nested structure:
    Country -> State -> City -> Station -> Pollutant_Index / Air_Quality_Index
    """

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    records: list[dict[str, Any]] = []
    for country in root.findall("Country"):
        for state in country.findall("State"):
            for city in state.findall("City"):
                for station in city.findall("Station"):
                    rec: dict[str, Any] = {
                        "state": state.get("id"),
                        "city": city.get("id"),
                        "station": station.get("id"),
                    }

                    last_update = station.get("lastupdate") or ""
                    parts = last_update.split()
                    if len(parts) >= 2:
                        rec["date"] = parts[0]
                        rec["time"] = parts[1]
                    else:
                        rec["date"] = None
                        rec["time"] = None

                    # Pollutants
                    for pindex in station.findall("Pollutant_Index"):
                        pid = (pindex.get("id") or "").strip()
                        avg = pindex.get("Avg")
                        if pid:
                            rec[pid] = avg

                    aqi_node = station.find("Air_Quality_Index")
                    if aqi_node is not None:
                        rec["AQI"] = aqi_node.get("Value")
                        rec["Predominant_Parameter"] = aqi_node.get("Predominant_Parameter")
                    else:
                        rec["AQI"] = None
                        rec["Predominant_Parameter"] = None

                    records.append(rec)

    return pd.DataFrame.from_records(records)


def xml_to_csv(xml_path: Path, *, output_csv: Path | None = None) -> XmlToCsvResult:
    """Convert an AQI XML file into a CSV file."""

    df = parse_cpcb_xml(xml_path)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return XmlToCsvResult(rows=int(len(df)), output_csv=output_csv)

