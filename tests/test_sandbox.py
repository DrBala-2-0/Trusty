"""
Standalone sandbox smoke test — Chapter 17.
Run from project root:  python tests/test_sandbox.py
Does NOT require the FastAPI server to be running.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.sandbox_client import run

TESTS = [
    {
        "name": "basic stdout",
        "code": "print('hello from sandbox')",
        "expect_stdout": "hello from sandbox",
        "expect_chart": False,
        "expect_error": False,
    },
    {
        "name": "pandas dataframe",
        "code": (
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})\n"
            "print(df['a'].sum())"
        ),
        "expect_stdout": "6",
        "expect_chart": False,
        "expect_error": False,
    },
    {
        "name": "matplotlib chart",
        "code": (
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1,2,3],[4,5,6])\n"
            "plt.title('Test')"
        ),
        "expect_stdout": "",
        "expect_chart": True,
        "expect_error": False,
    },
    {
        "name": "timeout enforcement",
        "code": "import time\ntime.sleep(999)",
        "expect_stdout": "",
        "expect_chart": False,
        "expect_error": True,   # expects result.error to be set
    },
    {
        "name": "no network access",
        "code": (
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://example.com', timeout=3)\n"
            "    print('NETWORK_ALLOWED')\n"
            "except Exception as e:\n"
            "    print('NETWORK_BLOCKED')"
        ),
        "expect_stdout": "NETWORK_BLOCKED",
        "expect_chart": False,
        "expect_error": False,
    },
]

passed = 0
failed = 0

for t in TESTS:
    result = run(t["code"], backend="docker")
    ok = True
    notes = []

    if t["expect_error"]:
        if not result.error:
            ok = False
            notes.append("expected error but got none")
    else:
        if result.error:
            ok = False
            notes.append(f"unexpected error: {result.error}")
        if t["expect_stdout"] and t["expect_stdout"] not in result.stdout:
            ok = False
            notes.append(f"stdout mismatch: {result.stdout!r}")
        if t["expect_chart"] and not result.chart_b64:
            ok = False
            notes.append("expected chart_b64 but got None")

    status = "[OK]" if ok else "[FAIL]"
    print(f"{status} {t['name']}"
          + (f" -- {'; '.join(notes)}" if notes else "")
          + f" ({result.elapsed_seconds:.2f}s)")

    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{passed}/{passed+failed} passed")
if failed:
    sys.exit(1)