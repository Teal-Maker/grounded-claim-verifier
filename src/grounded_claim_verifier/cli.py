"""Command-line interface for grounded-claim-verifier."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from grounded_claim_verifier.extractor import extract_claims
from grounded_claim_verifier.verifier import ClaimVerifier, VerifierConfig


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def sample_claims(
    temp_claims: list[dict[str, Any]],
    pressure_claims: list[dict[str, Any]],
    n_temp: int = 25,
    n_pressure: int = 25,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to ``n_temp`` temperature and ``n_pressure`` pressure claims.

    Falls back to the full available count when fewer claims exist.

    Parameters
    ----------
    temp_claims:
        All temperature claims extracted from the dataset.
    pressure_claims:
        All pressure claims extracted from the dataset.
    n_temp:
        Maximum number of temperature claims to sample.
    n_pressure:
        Maximum number of pressure claims to sample.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Combined sample with a ``claim_type`` key (``"temperature"`` or
        ``"pressure"``) added to each claim.
    """
    rng = random.Random(seed)

    sampled_temp = [
        {**c, "claim_type": "temperature"}
        for c in rng.sample(temp_claims, min(n_temp, len(temp_claims)))
    ]
    sampled_pressure = [
        {**c, "claim_type": "pressure"}
        for c in rng.sample(pressure_claims, min(n_pressure, len(pressure_claims)))
    ]

    return sampled_temp + sampled_pressure


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the grounded claim verification pipeline.

    Loads QA records from a JSONL input file, extracts and samples numeric
    claims, fetches source texts via the selected provider, verifies each
    claim, and writes a JSON results file with per-claim verdicts and an
    overall hallucination rate.
    """
    parser = argparse.ArgumentParser(
        description="Source-grounded numeric claim verification for QA datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file containing QA records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spot_check_results.json"),
        help="Output JSON results file",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="JSONL file of source texts ({id, text} per line) — mutually exclusive with --db-url",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL connection string for DatabaseProvider — falls back to DATABASE_URL env var",
    )
    parser.add_argument(
        "--db-table",
        type=str,
        default="sources",
        help="Table name when using --db-url",
    )
    parser.add_argument(
        "--db-id-column",
        type=str,
        default="id",
        help="ID column when using --db-url",
    )
    parser.add_argument(
        "--db-text-column",
        type=str,
        default="text",
        help="Text column when using --db-url",
    )
    parser.add_argument(
        "--n-temp",
        type=int,
        default=25,
        help="Number of temperature claims to sample",
    )
    parser.add_argument(
        "--n-pressure",
        type=int,
        default=25,
        help="Number of pressure claims to sample",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Numeric tolerance for approximate matching (0.10 = ±10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for claim sampling",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load records
    # ------------------------------------------------------------------
    print(f"[Verifier] Loading {args.input} ...")
    records: list[dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                records.append(json.loads(raw))
    print(f"  Loaded {len(records):,} records")

    # ------------------------------------------------------------------
    # 2. Extract claims
    # ------------------------------------------------------------------
    print("[Verifier] Extracting numeric claims ...")
    temp_claims, pressure_claims = extract_claims(records)
    print(f"  Temperature claims: {len(temp_claims):,}")
    print(f"  Pressure claims:    {len(pressure_claims):,}")

    if not temp_claims and not pressure_claims:
        print("[Verifier] No claims found — nothing to verify.")
        return

    # ------------------------------------------------------------------
    # 3. Sample claims
    # ------------------------------------------------------------------
    sampled = sample_claims(
        temp_claims, pressure_claims, args.n_temp, args.n_pressure, args.seed
    )
    n_sampled_temp = sum(1 for c in sampled if c["claim_type"] == "temperature")
    n_sampled_pres = sum(1 for c in sampled if c["claim_type"] == "pressure")
    print(
        f"  Sampled {len(sampled)} claims "
        f"({n_sampled_temp} temp, {n_sampled_pres} pressure)"
    )

    # ------------------------------------------------------------------
    # 4. Build source provider
    # ------------------------------------------------------------------
    if args.sources and args.db_url:
        print("[ERROR] --sources and --db-url are mutually exclusive.")
        sys.exit(1)

    if args.sources:
        from grounded_claim_verifier.providers.jsonl import JSONLProvider

        provider = JSONLProvider(args.sources)
        print(f"  Source provider: JSONLProvider({args.sources})")

    else:
        # Database provider — resolve connection string.
        db_url = args.db_url or os.environ.get("DATABASE_URL")
        if not db_url:
            print(
                "[ERROR] No source provider configured.\n"
                "  Use --sources <file.jsonl>  or\n"
                "       --db-url <connection_string>  or\n"
                "  set the DATABASE_URL environment variable."
            )
            sys.exit(1)

        from grounded_claim_verifier.providers.database import DatabaseProvider

        provider = DatabaseProvider(  # type: ignore[assignment]
            connection_string=db_url,
            table_name=args.db_table,
            id_column=args.db_id_column,
            text_column=args.db_text_column,
        )
        print(f"  Source provider: DatabaseProvider (table={args.db_table})")

    # ------------------------------------------------------------------
    # 5. Verify claims
    # ------------------------------------------------------------------
    print(f"[Verifier] Verifying {len(sampled)} claims (tolerance ±{args.tolerance:.0%}) ...")
    config = VerifierConfig(tolerance=args.tolerance)
    verifier = ClaimVerifier(provider, config)
    results = verifier.verify_claims(sampled)

    # ------------------------------------------------------------------
    # 6. Tally verdicts
    # ------------------------------------------------------------------
    verdicts: dict[str, int] = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    confirmed = verdicts.get("confirmed", 0)
    confirmed_untyped = verdicts.get("confirmed_untyped", 0)
    approximate = verdicts.get("approximate", 0)
    in_range = verdicts.get("in_range", 0)
    not_found = verdicts.get("not_found", 0)
    no_source = verdicts.get("no_source", 0)

    verifiable = len(results) - no_source
    hallucination_count = not_found
    hallucination_rate = hallucination_count / max(verifiable, 1)

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    bar = "=" * 60
    print(f"\n{bar}")
    print("SPOT CHECK RESULTS")
    print(bar)
    print(f"  Total claims checked:     {len(results)}")
    print(f"  Confirmed (exact typed):  {confirmed}")
    print(f"  Confirmed (untyped ctx):  {confirmed_untyped}")
    print(f"  Approximate (±{args.tolerance:.0%}):      {approximate}")
    print(f"  In source range:          {in_range}")
    print(f"  Not found in source:      {not_found}")
    print(f"  No source available:      {no_source}")
    print()
    print(f"  Verifiable claims:        {verifiable}")
    print(
        f"  Hallucination rate:       "
        f"{hallucination_rate:.1%} ({hallucination_count}/{verifiable})"
    )
    print(bar)

    for ctype in ("temperature", "pressure"):
        type_results = [r for r in results if r.get("claim_type") == ctype]
        type_verifiable = sum(1 for r in type_results if r["verdict"] != "no_source")
        type_not_found = sum(1 for r in type_results if r["verdict"] == "not_found")
        type_rate = type_not_found / max(type_verifiable, 1)
        print(
            f"  {ctype.capitalize()}: "
            f"{type_not_found}/{type_verifiable} not found ({type_rate:.1%})"
        )

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    output_data: dict[str, Any] = {
        "input_file": str(args.input),
        "total_temp_claims": len(temp_claims),
        "total_pressure_claims": len(pressure_claims),
        "sample_size": len(results),
        "tolerance": args.tolerance,
        "seed": args.seed,
        "verdicts": verdicts,
        "hallucination_rate": round(hallucination_rate, 4),
        "verifiable_count": verifiable,
        "claims": results,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results written to: {args.output}")


if __name__ == "__main__":
    main()
