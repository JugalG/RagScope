"""Grounding check: does each claim really follow from the passages it cites?

The citation check in generate.py only proves the labels exist. Here a cheap model reads
each claim next to its cited passages and says supported / not supported. Unsupported
claims are removed from the final answer. This is a judgment by a model, so it can be
wrong, but it catches many invented details.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag import config, llm, trace
from rag.generate import Answer


class Verdict(BaseModel):
    claim_number: int
    supported: bool
    reason: str = Field(description="Short reason, max 15 words")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def check(answer: Answer, labeled: dict[str, dict]) -> tuple[Answer, list[dict], str]:
    """Returns (answer with only supported claims, per-claim report, status)."""
    if answer.abstained or not answer.claims:
        return answer, [], "not_needed"

    items = []
    for n, claim in enumerate(answer.claims, start=1):
        evidence = "\n".join(f"[{s}] {labeled[s]['text']}" for s in claim.sources)
        items.append(f"CLAIM {n}: {claim.text}\nCITED PASSAGES:\n{evidence}")
    try:
        result = llm.cheap_structured(Verdicts).invoke(
            [
                SystemMessage(
                    "For each claim, decide if it is fully supported by ITS cited passages alone. "
                    "Paraphrase is fine; added facts, numbers or names not in the passages are not. "
                    "Return one verdict per claim."
                ),
                HumanMessage("\n\n".join(items)),
            ]
        )
    except Exception as exc:
        trace.warn(f"Grounding check unavailable ({type(exc).__name__}); keeping claims unverified.")
        return answer, [], "unverified"

    by_number = {v.claim_number: v for v in result.verdicts}
    report, kept = [], []
    for n, claim in enumerate(answer.claims, start=1):
        v = by_number.get(n)
        ok = bool(v and v.supported)
        report.append({"claim": claim.text, "supported": ok, "reason": v.reason if v else "no verdict"})
        if ok:
            kept.append(claim)

    trace.table(
        f"Grounding check (judge: {config.CHEAP_MODEL})",
        ["#", "supported?", "reason", "claim"],
        [
            [n, "yes" if r["supported"] else "[red]NO[/]", trace.short(r["reason"], 60), trace.short(r["claim"], 50)]
            for n, r in enumerate(report, start=1)
        ],
    )
    if not kept:
        return Answer(claims=[], abstained=True), report, "failed"
    status = "grounded" if len(kept) == len(answer.claims) else "partially_grounded"
    return Answer(claims=kept, abstained=False), report, status
