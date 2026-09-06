"""Assess how fresh/current a medical evidence date is.

Converts a date string (e.g. "2023-05-15", "2019", "N/A") into a
freshness category based on its age relative to today.

Run: python freshness.py
"""

import re
from datetime import date

CATEGORIES = {
    "FRESH": 2,
    "RECENT": 5,
    "AGING": 10,
    "OUTDATED": float("inf"),
}


def freshness_assessment(date_str, today=None):
    today = today or date.today()

    s = str(date_str).strip() if date_str is not None else ""
    if not s or s.upper() in ("N/A", "NA", "UNKNOWN", "NONE", "NULL"):
        return {
            "input": s,
            "parsed": None,
            "age_years": None,
            "category": "UNKNOWN",
            "is_current": False,
        }

    match = re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", s)
    if not match:
        return {
            "input": s,
            "parsed": None,
            "age_years": None,
            "category": "UNKNOWN",
            "is_current": False,
        }

    year, month, day = (int(g) if g else None for g in match.groups())
    try:
        parsed = date(year, month or 1, day or 1)
    except ValueError:
        return {
            "input": s,
            "parsed": None,
            "age_years": None,
            "category": "UNKNOWN",
            "is_current": False,
        }

    age_years = (today - parsed).days / 365.25
    category = next(
        name for name, limit in CATEGORIES.items() if age_years < limit
    )

    return {
        "input": s,
        "parsed": parsed.isoformat(),
        "age_years": round(age_years, 1),
        "category": category,
        "is_current": age_years < 2,
    }


def main():
    test_dates = ["2026", "2023", "2019", "2015", "N/A"]

    for test_date in test_dates:
        result = freshness_assessment(test_date)
        print(f"\nDate: {test_date!r}")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()