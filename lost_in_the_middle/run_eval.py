#!/usr/bin/env python3
"""
run_eval.py – Evaluate the "Lost in the Middle" position effect.

Usage:
    python -m lost_in_the_middle.run_eval \
        --model gpt-4o-mini \
        --trials 10 \
        --context-lens 10 30 60 \
        --positions start early middle late end \
        --out results.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys

from openai import AsyncOpenAI

from lost_in_the_middle.data import build_context

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "Reply with ONLY the answer — no explanation, no extra words."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (short, exact phrase only):"
)


def normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[.,;:!?'\"-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized containment check."""
    return normalize(gold) in normalize(prediction)


async def query_model(client, model: str, context: str, question: str, sem) -> str:
    user_msg = USER_TEMPLATE.format(context=context, question=question)
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=64,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    return resp.choices[0].message.content.strip()


async def run_one(client, model, sem, sample):
    answer = await query_model(client, model, sample["context"], sample["question"], sem)
    correct = exact_match(answer, sample["gold_answer"])
    return {
        "context_len": sample["context_len"],
        "needle_pos": sample["needle_pos"],
        "trial_id": sample["trial_id"],
        "correct": correct,
        "model_answer": answer,
        "gold_answer": sample["gold_answer"],
    }


async def amain(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(15)

    tasks = []
    for ctx_len in args.context_lens:
        for pos in args.positions:
            for t in range(args.trials):
                sample = build_context(
                    num_distractor_sentences=ctx_len,
                    needle_position=pos,
                    trial_id=t,
                    seed=args.seed,
                )
                tasks.append(run_one(client, args.model, sem, sample))

    print(f"Running {len(tasks)} evaluations...")
    results = await asyncio.gather(*tasks)

    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} ctx={r['context_len']:>3d} pos={r['needle_pos']:<8s} trial={r['trial_id']} "
              f"gold={r['gold_answer']!r}  ans={r['model_answer']!r}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["context_len"], r["needle_pos"])].append(r["correct"])
    print(f"\n{'ctx_len':>8s} {'position':<10s} {'accuracy':>8s} {'n':>4s}")
    print("-" * 35)
    for (cl, pos), vals in sorted(groups.items()):
        acc = sum(vals) / len(vals) * 100
        print(f"{cl:>8d} {pos:<10s} {acc:>7.1f}% {len(vals):>4d}")

    print(f"\n✅  Wrote {len(results)} records to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Lost-in-the-Middle eval harness")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--context-lens", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument("--positions", type=str, nargs="+",
                        default=["start", "early", "middle", "late", "end"])
    parser.add_argument("--out", type=str, default="results.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
