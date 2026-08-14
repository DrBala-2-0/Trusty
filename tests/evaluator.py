"""
Trusty Chapter 10 — Golden Set Evaluator

Runs every case in tests/golden_set.json against a live Trusty server
(default: http://localhost:8000) and prints a structured pass/fail report.

Usage:
    python tests/evaluator.py                        # default server
    python tests/evaluator.py --base-url http://localhost:8001

Prerequisites:
    - Server must be running: uvicorn app:app --reload
    - facts.txt, offtopic.txt, injection.txt must exist in tests/fixtures/
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import requests

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_SET = Path(__file__).parent / "golden_set.json"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def upload_fixture(base_url: str, fixture_name: str, session_id: str) -> None:
    """Upload a fixture file to /upload under a dedicated session."""
    fixture_path = FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    with open(fixture_path, "rb") as f:
        response = requests.post(
            f"{base_url}/upload",
            files={"file": (fixture_name, f, "text/plain")},
            headers={"X-Session-ID": session_id},
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upload failed for {fixture_name} "
            f"(HTTP {response.status_code}): {response.text}"
        )


def ask_question(base_url: str, question: str, session_id: str) -> dict:
    """POST a question to /ask and return the parsed JSON response."""
    response = requests.post(
        f"{base_url}/ask",
        json={"text": question},
        headers={"X-Session-ID": session_id},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ask failed (HTTP {response.status_code}): {response.text}"
        )

    return response.json()


# ---------------------------------------------------------------------------
# Pass/fail logic
# ---------------------------------------------------------------------------

def evaluate_case(result: dict, criteria: dict) -> tuple[bool, list[str]]:
    """
    Check a single case's result against its pass_criteria.
    Returns (passed: bool, reasons: list[str]).
    reasons is empty on a full pass; contains one line per failed check.
    """
    failures = []
    draft = result.get("draft_answer", "").lower()
    parsed = result.get("parsed_report", {})

    # Supported / Relevant fields (only checked when criteria specifies them)
    if "supported" in criteria:
        actual = parsed.get("Supported", "").strip().upper()
        expected = criteria["supported"].upper()
        if actual != expected:
            failures.append(
                f"Supported: expected {expected}, got '{actual}'"
            )

    if "relevant" in criteria:
        actual = parsed.get("Relevant", "").strip().upper()
        expected = criteria["relevant"].upper()
        if actual != expected:
            failures.append(
                f"Relevant: expected {expected}, got '{actual}'"
            )

    # Answer content checks (case-insensitive)
    for phrase in criteria.get("answer_must_contain", []):
        if phrase.lower() not in draft:
            failures.append(f"Answer missing expected phrase: '{phrase}'")

    for phrase in criteria.get("answer_must_not_contain", []):
        if phrase.lower() in draft:
            failures.append(f"Answer contains forbidden phrase: '{phrase}'")

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(base_url: str) -> int:
    """
    Run all cases in golden_set.json.
    Returns exit code: 0 if all passed, 1 if any failed.
    """
    with open(GOLDEN_SET) as f:
        golden = json.load(f)

    version = golden.get("version", "unknown")
    cases = golden["cases"]

    print(f"\n{'=' * 60}")
    print(f"  Trusty Golden Set Evaluator  (v{version})")
    print(f"  Server : {base_url}")
    print(f"  Cases  : {len(cases)}")
    print(f"{'=' * 60}\n")

    # Group cases by fixture so we only upload each fixture once.
    # Each fixture gets its own session_id so sessions don't bleed
    # into each other — the same isolation Chapter 7 tested manually.
    fixture_sessions: dict[str, str] = {}

    results = []
    for case in cases:
        case_id = case["id"]
        fixture = case["fixture"]
        question = case["question"]
        criteria = case["pass_criteria"]

        print(f"[{case_id}] {case['description']}")
        print(f"         Fixture  : {fixture}")
        print(f"         Question : {question}")

        # Upload fixture once per fixture name, reuse session after that.
        if fixture not in fixture_sessions:
            session_id = uuid.uuid4().hex
            fixture_sessions[fixture] = session_id
            try:
                upload_fixture(base_url, fixture, session_id)
                print(f"         Upload   : OK (session {session_id[:8]}...)")
            except Exception as e:
                print(f"         Upload   : FAILED — {e}")
                print(f"         Result   : SKIP (upload error)\n")
                results.append((case_id, False, [f"Upload error: {e}"]))
                continue
        else:
            session_id = fixture_sessions[fixture]
            print(f"         Upload   : reusing session {session_id[:8]}...")

        # Ask the question.
        try:
            result = ask_question(base_url, question, session_id)
        except Exception as e:
            print(f"         Ask      : FAILED — {e}")
            print(f"         Result   : SKIP (ask error)\n")
            results.append((case_id, False, [f"Ask error: {e}"]))
            continue

        draft = result.get("draft_answer", "")
        parsed = result.get("parsed_report", {})
        supported = parsed.get("Supported", "—")
        relevant = parsed.get("Relevant", "—")

        print(f"         Answer   : {draft[:120]}{'...' if len(draft) > 120 else ''}")
        print(f"         Verified : Supported={supported}  Relevant={relevant}")

        passed, failures = evaluate_case(result, criteria)

        if passed:
            print(f"         Result   : PASS\n")
        else:
            print(f"         Result   : FAIL")
            for reason in failures:
                print(f"                    → {reason}")
            print()

        results.append((case_id, passed, failures))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = total - passed_count

    print(f"{'=' * 60}")
    print(f"  SUMMARY   {passed_count}/{total} passed")
    print(f"{'=' * 60}")

    if failed_count > 0:
        print(f"\n  Failed cases:")
        for case_id, passed, failures in results:
            if not passed:
                print(f"    [{case_id}]")
                for reason in failures:
                    print(f"      → {reason}")
        print()
        return 1

    print(f"\n  All {total} cases passed — golden set gate cleared.\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trusty golden set evaluator")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running Trusty server (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    sys.exit(run_evaluation(args.base_url))