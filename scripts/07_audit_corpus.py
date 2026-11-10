
"""Audit Wikipedia crawler corpus, rejected candidates and request efficiency."""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter

from wikimedia_common import DB_PATH, DATA_DIR


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def reason_group(error):
    if not error:
        return "Unspecified"

    if error.startswith("Too short:"):
        return "Too short"

    if error.startswith("Page ID mismatch"):
        return "Page ID mismatch / redirect"

    if error.startswith("Missing, non-article"):
        return "Missing / namespace / redirect"

    if error.startswith("Disambiguation"):
        return "Disambiguation"

    if error.startswith("List/outline"):
        return "List / outline"

    return error.split(":", 1)[0][:90]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=15)
    parser.add_argument("--min-words", type=int, default=300)
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"No database found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:

        print("Database:", DB_PATH)
        print("Data root:", DATA_DIR)

        # ------------------------------------------
        # 1. TOTAL DOCUMENTS
        # ------------------------------------------

        total = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        print("\nTotal unique documents:", total)

        # ------------------------------------------
        # 2. DISTRIBUTION BY AREA
        # ------------------------------------------

        print("\nBy source area:")

        for area, n in conn.execute(
            """
            SELECT area, COUNT(*)
            FROM documents
            GROUP BY area
            ORDER BY COUNT(*) DESC, area
            """
        ):
            print(f"{area:27} {n:6,}")

        # ------------------------------------------
        # 3. DOCUMENT LENGTHS
        # ------------------------------------------

        print("\nDocument lengths (min / mean / max):")

        print(
            conn.execute(
                """
                SELECT
                    MIN(word_count),
                    ROUND(AVG(word_count), 1),
                    MAX(word_count)
                FROM documents
                """
            ).fetchone()
        )

        short_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE word_count < ?",
            (args.min_words,)
        ).fetchone()[0]

        print(
            f"\nDocuments shorter than {args.min_words} words:",
            short_count
        )

        print("\nShortest documents:")

        for row in conn.execute(
            """
            SELECT pageid, title, area, word_count
            FROM documents
            ORDER BY word_count ASC
            LIMIT ?
            """,
            (args.examples,)
        ):
            print(row)

        print("\nLongest documents:")

        for row in conn.execute(
            """
            SELECT pageid, title, area, word_count
            FROM documents
            ORDER BY word_count DESC
            LIMIT ?
            """,
            (min(args.examples, 10),)
        ):
            print(row)

        # ------------------------------------------
        # 4. CANDIDATE STATES
        # ------------------------------------------

        print("\nCandidate states:")

        states = dict(
            conn.execute(
                """
                SELECT status, COUNT(*)
                FROM candidates
                GROUP BY status
                """
            )
        )

        for status, count in sorted(states.items()):
            print(f"  {status:17} {count:,}")

        reviewed = (
            states.get("completed", 0)
            + states.get("rejected", 0)
        )

        if reviewed:

            accepted_fraction = (
                states.get("completed", 0) / reviewed
            )

            print(
                "Accepted fraction among completed/rejected candidates:",
                f"{accepted_fraction:.1%}"
            )

        # ------------------------------------------
        # 5. REJECTION REASONS
        # ------------------------------------------

        print("\nRejection reasons (new crawler):")

        reasons = Counter()

        for reason, count in conn.execute(
            """
            SELECT last_error, COUNT(*)
            FROM candidates
            WHERE status = 'rejected'
            GROUP BY last_error
            """
        ):
            reasons[reason_group(reason)] += count

        for reason, count in reasons.most_common():
            print(f"  {reason:45} {count:,}")

        if not reasons:
            print("  None")

        # ------------------------------------------
        # 6. LEGACY DOCUMENT VALIDATION
        # ------------------------------------------

        print("\nLegacy document validation:")

        if table_exists(conn, "legacy_review"):

            for status, count in conn.execute(
                """
                SELECT status, COUNT(*)
                FROM legacy_review
                GROUP BY status
                ORDER BY status
                """
            ):
                print(f"  {status:17} {count:,}")

            for pid, status, reason in conn.execute(
                """
                SELECT pageid, status, reason
                FROM legacy_review
                WHERE status != 'accepted'
                ORDER BY status, pageid
                LIMIT ?
                """,
                (args.examples,)
            ):
                print(f"  {pid} [{status}] {reason}")

        else:
            print(
                "  No legacy review table "
                "(run the updated crawler first)."
            )

        # ------------------------------------------
        # 7. HTTP REQUEST COUNTERS
        # ------------------------------------------

        print(
            "\nHTTP requests by purpose "
            "(since counters were enabled):"
        )

        if table_exists(conn, "stats"):

            entries = conn.execute(
                """
                SELECT key, value
                FROM stats
                WHERE key LIKE 'http_total_%'
                ORDER BY key
                """
            ).fetchall()

            for key, value in entries:
                purpose = key.removeprefix("http_total_")
                print(f"  {purpose:25} {value}")

            if not entries:
                print(
                    "  No measurements yet "
                    "(run the updated crawler)."
                )

        # ------------------------------------------
        # 8. REMAINING CATEGORIES
        # ------------------------------------------

        print("\nTopic categories remaining:")

        remaining = conn.execute(
            """
            SELECT COUNT(*)
            FROM topics
            WHERE done = 0
            """
        ).fetchone()[0]

        print(remaining)

        # ------------------------------------------
        # 9. VERIFY UNIQUE PAGE IDS
        # ------------------------------------------

        unique_ids = conn.execute(
            """
            SELECT COUNT(DISTINCT pageid)
            FROM documents
            """
        ).fetchone()[0]

        print("\nUnique IDs in database:", unique_ids)


if __name__ == "__main__":
    main()
