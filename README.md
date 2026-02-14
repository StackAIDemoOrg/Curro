# Curro – Lost in the Middle Replication

A minimal replication of the **"Lost in the Middle"** position effect
([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) using synthetic data
and the OpenAI API.

## Key Finding

Language models exhibit a **U-shaped accuracy curve** when a relevant fact
(needle) is embedded among distractor sentences (haystack). Performance is
highest when the needle is at the **start** or **end** of the context, and
drops when it is in the **middle** — even for models with large context windows.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

### 1. Run the evaluation

```bash
python -m lost_in_the_middle.run_eval \
    --model gpt-4o-mini \
    --trials 5 \
    --context-lens 10 20 40 \
    --positions start middle end \
    --out results.jsonl
```

| Flag | Description |
|------|-------------|
| `--model` | OpenAI model name (default: `gpt-4o-mini`) |
| `--trials` | Number of independent trials per condition (default: 5) |
| `--context-lens` | List of distractor-sentence counts to test |
| `--positions` | Needle positions: `start`, `middle`, `end` |
| `--out` | Output JSONL file path |
| `--seed` | Random seed for reproducibility (default: 42) |

### 2. Plot results

```bash
python -m lost_in_the_middle.plot_results results.jsonl --out plot.png
```

## Output Format (JSONL)

Each line is a JSON object:

```json
{
  "context_len": 20,
  "needle_pos": "middle",
  "trial_id": 0,
  "correct": false,
  "model_answer": "...",
  "gold_answer": "Luminara"
}
```

## How It Works

1. **Synthetic data**: A pool of 5 needle facts (fictional, unambiguous) and
   50 distractor sentences. For each trial the needle is placed at the
   start / middle / end of a block of N distractor sentences.
2. **Prompt**: A system message instructs the model to answer with only the
   exact phrase; `temperature=0` for determinism.
3. **Evaluation**: Case-insensitive substring match (gold answer ∈ model
   answer).
4. **Plotting**: Grouped bar chart of accuracy vs. needle position, one
   group per context length.

## Reference

> Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
> Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long
> Contexts.* TACL 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
