# grounded-claim-verifier

Source-grounded numeric claim verification for synthetic training data.

## Why use this?

LLMs hallucinate numbers. When you generate synthetic QA pairs from source documents, the answers often contain temperatures, pressures, or other measurements that look plausible but don't appear anywhere in the source material. Standard hallucination detectors (like SelfCheckGPT) use sampling-based consistency checks, but they can't tell you whether a specific number is actually grounded in the original text.

This library extracts numeric claims from LLM-generated text and checks each one against the source document using layered matching: exact value, approximate match within a tolerance, range containment (e.g. "75 psig" found in a source range of "60-80 psig"), and an untyped fallback for transcripts that drop unit symbols.

**Use this when you are:**
- Validating synthetic training data generated from source documents
- Spot-checking hallucination rates in LLM-generated technical content
- Auditing numeric accuracy in QA pairs before fine-tuning

## Installation

```bash
pip install grounded-claim-verifier

# With PostgreSQL support:
pip install "grounded-claim-verifier[database]"
```

## Input Format

**QA records** (`qa_dataset.jsonl`) — one JSON object per line:

```json
{"qa_id": "q001", "source_content_id": "src_42", "answer": "The suction pressure should be around 70 psig..."}
```

For multi-turn records, use `"type": "multi_turn"` with a `"conversation"` array instead of `"answer"`.

**Source texts** (`sources.jsonl`) — one JSON object per line:

```json
{"id": "src_42", "text": "Normal suction pressure for R-410A is 60-80 psig at typical conditions..."}
```

## Quick Start

```python
from grounded_claim_verifier import ClaimVerifier, JSONLProvider, extract_claims

# Load QA records
import json
records = [json.loads(line) for line in open("qa_dataset.jsonl")]

# Extract claims
temp_claims, pressure_claims = extract_claims(records)

# Verify against source texts
provider = JSONLProvider("sources.jsonl")
verifier = ClaimVerifier(provider)
results = verifier.verify_claims(temp_claims + pressure_claims)

# Each result has: verdict, source_match, source_context, match_type
for r in results:
    print(r["verdict"], r["value"], r["unit"])
```

## Verdicts

| Verdict | Meaning |
|---------|---------|
| `confirmed` | Exact typed match in source |
| `approximate` | Within configured tolerance (default ±10%) |
| `in_range` | Value falls within a source range (e.g. "60-80 psig") |
| `confirmed_untyped` | Bare number + measurement-adjacent keyword in source |
| `not_found` | No supporting evidence in source |
| `no_source` | Source text unavailable for this claim |

## CLI

```bash
grounded-claim-verifier \
    --input qa_dataset.jsonl \
    --sources sources.jsonl \
    --output results.json \
    --n-temp 25 \
    --n-pressure 25 \
    --tolerance 0.10
```

## License

Apache-2.0
