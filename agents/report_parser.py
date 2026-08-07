from utils.logging import logger

CANONICAL_KEYS = ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]
_KEY_LOOKUP = {key.lower(): key for key in CANONICAL_KEYS}


def parse_verification_report(report_text: str) -> dict:
    """Parse the verification agent's free-text report into a structured dict.

    Keys are matched case-insensitively against CANONICAL_KEYS. This exists because
    LLM output formatting isn't guaranteed byte-for-byte consistent across calls —
    an exact-match parser silently drops any line whose casing doesn't match, with
    no error and no visible sign anything was lost.
    """
    result = {key: "" for key in CANONICAL_KEYS}

    for line in report_text.splitlines():
        if ":" not in line:
            continue
        raw_key, _, value = line.partition(":")
        canonical = _KEY_LOOKUP.get(raw_key.strip().lower())
        if canonical:
            result[canonical] = value.strip()
        else:
            logger.debug(f"Unrecognized verification report line, dropped: {line!r}")

    return result