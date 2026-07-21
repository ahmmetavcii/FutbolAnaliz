#!/usr/bin/env python3
"""Run the six SoccerNet capability regressions and emit machine reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "soccernet_repo_tests" / "regression"


@dataclass
class Result:
    component: str
    passed: bool
    duration_s: float
    detail: str


def _pytest(component: str, test_file: str) -> Result:
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test_file],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout.strip()
    return Result(
        component=component,
        passed=process.returncode == 0,
        duration_s=time.perf_counter() - started,
        detail=output[-2000:],
    )


def _sdk_regression() -> Result:
    started = time.perf_counter()
    path = ROOT / "artifacts/soccernet_repo_tests/SoccerNet/sdk_regression.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        passed = (
            payload["technical_status"] == "PASS"
            and payload["downloader_no_network"] is True
            and payload["sample_decode_ok"] == payload["sample_decode_total"] == 100
            and payload["zero_byte_count"] == 0
        )
        detail = json.dumps(payload, sort_keys=True)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        passed, detail = False, str(exc)
    return Result("SoccerNet SDK", passed, time.perf_counter() - started, detail)


def _write_reports(results: list[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(result.passed for result in results)
    payload = {
        "technical_status": "PASS" if passed == len(results) else "FAIL",
        "tests_passed": passed,
        "tests_failed": len(results) - passed,
        "components": [asdict(result) for result in results],
    }
    (output_dir / "soccernet_regression.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SoccerNet component regression",
        "",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        "",
        "| Component | Status | Duration (s) |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {result.component} | {'PASS' if result.passed else 'FAIL'} | "
        f"{result.duration_s:.2f} |"
        for result in results
    )
    (output_dir / "soccernet_regression.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    suite = ET.Element(
        "testsuite",
        name="soccernet-components",
        tests=str(len(results)),
        failures=str(len(results) - passed),
        time=f"{sum(result.duration_s for result in results):.3f}",
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname="soccernet",
            name=result.component,
            time=f"{result.duration_s:.3f}",
        )
        if not result.passed:
            failure = ET.SubElement(case, "failure", message="component regression failed")
            failure.text = result.detail
        output = ET.SubElement(case, "system-out")
        output.text = result.detail
    ET.ElementTree(suite).write(
        output_dir / "soccernet_regression.junit.xml",
        encoding="utf-8",
        xml_declaration=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = [
        _sdk_regression(),
        _pytest("sn-trackeval", "tests/test_trackeval_adapter.py"),
        _pytest("sn-echoes", "tests/test_sn_echoes_reader.py"),
        _pytest("sn-calibration", "tests/test_sn_calibration_compatible.py"),
        _pytest("sn-jersey", "tests/test_jersey_recognition.py"),
        _pytest("sn-gamestate", "tests/test_sn_gamestate_compatible.py"),
    ]
    _write_reports(results, args.output_dir)
    print(json.dumps({result.component: result.passed for result in results}, indent=2))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
