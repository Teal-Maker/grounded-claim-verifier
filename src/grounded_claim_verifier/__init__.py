"""grounded-claim-verifier — source-grounded numeric claim verification.

Quick start::

    from grounded_claim_verifier import ClaimVerifier, JSONLProvider, extract_claims

    provider = JSONLProvider("sources.jsonl")
    verifier = ClaimVerifier(provider)

    records = [...]                          # list of QA record dicts
    temp_claims, pressure_claims = extract_claims(records)
    results = verifier.verify_claims(temp_claims + pressure_claims)

See the project README for full documentation.
"""

from grounded_claim_verifier.extractor import extract_claims
from grounded_claim_verifier.providers import DatabaseProvider, JSONLProvider, SourceProvider
from grounded_claim_verifier.verifier import ClaimVerifier, VerifierConfig, verify_claim

__all__ = [
    # Core classes
    "ClaimVerifier",
    "VerifierConfig",
    # Providers
    "SourceProvider",
    "JSONLProvider",
    "DatabaseProvider",
    # Functions
    "extract_claims",
    "verify_claim",
]

__version__ = "0.1.0"
