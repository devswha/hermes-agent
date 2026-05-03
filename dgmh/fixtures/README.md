# DGM-H Judge Calibration Fixtures

Reference: devswha/dgmh @ 3eddf82a661688f88b4bbf4c55028509d0eac002
Plan: hermes-skill-archive-dgmh-plan-v2-addendum.md §R-1 (Codex-judge circularity calibration)

## Purpose

`judge-calibration.jsonl` holds 10 skill-task pairs sampled from the bundled
`/skills/` tree (mix of categories). These pairs anchor the W1 go-criterion:

> Codex-judge mean Spearman correlation with human ratings >= 0.7
> AND inter-reseed variance <= 0.2 over all fixture pairs.

If calibration fails (correlation < 0.7 or variance > 0.2), proceed to W1
no-go: stop, document in `dgmh/archive/gen-NNNN/operator-decision.md`,
revisit scoring source — leading alternative is held-out task-replay scorer
(Phase 2 candidate B from v2 §BLOCKER 2).

## Schema

Each line is a JSON object:

```json
{
  "skill_id":              "<category>/<name>",
  "task_prompt":           "The task given to the judge",
  "expected_output_summary": "What a high-quality response covers",
  "human_rating":          null,
  "codex_ratings":         []
}
```

## Operator fill-in process

1. Read each entry in `judge-calibration.jsonl`.
2. For each `skill_id`, open the corresponding SKILL.md under
   `~/.hermes/skills/<category>/<name>/SKILL.md`.
3. Mentally simulate what an agent guided by that skill would produce for
   `task_prompt`. Compare against `expected_output_summary`.
4. Assign a `human_rating` in [0.0, 1.0]:
   - 0.0 = skill is useless for this task
   - 0.5 = skill partially helps
   - 1.0 = skill fully guides a correct and complete response
5. Write your rating back into the JSONL field `human_rating`.

Example after rating:
```json
{"skill_id": "software-development/systematic-debugging", ..., "human_rating": 0.9, "codex_ratings": []}
```

6. Run the calibration script (W1 deliverable):
   ```
   python -m dgmh.calibrate --fixture dgmh/fixtures/judge-calibration.jsonl --reseeds 3
   ```
   This populates `codex_ratings` with `[seed0_score, seed1_score, seed2_score]`
   and prints the Spearman correlation + inter-reseed variance.

7. Gate decision:
   - Spearman >= 0.7 AND variance <= 0.2 → W1 go, proceed to W2
   - Otherwise → W1 no-go, document and revisit scoring source

## Coverage

The 10 entries span these categories:

| skill_id | category |
|---|---|
| mlops/huggingface-hub | mlops |
| software-development/systematic-debugging | software-development |
| software-development/test-driven-development | software-development |
| devops/webhook-subscriptions | devops |
| github/codebase-inspection | github |
| research/arxiv | research |
| mcp/native-mcp | mcp |
| apple/apple-notes | apple |
| research/research-paper-writing | research |
| social-media/xurl | social-media |

## Fixture update policy

Do NOT update fixtures to match changed Codex output. Update only when:
- The expected DGM-H contract intentionally changes, OR
- The original fixture is proven to encode the reference incorrectly.

Any update must record the old/new expected behavior and the reason.
See v1 plan §Reference Drift Prevention and v2 §R-1.
