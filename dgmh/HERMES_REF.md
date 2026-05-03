# DGM-H Hermes Reference Pin

This file pins the baseline coordinates used for the DGM-H Python port.
It is an audit coordinate, not a live dependency — see v2 addendum §Reference pin.

## Hermes-agent baseline

- **Repo**: devswha/hermes-agent
- **Branch pinned from**: main
- **HEAD SHA**: 6051fba9dc326ceddbe81147a14b10102f4256a3
- **Date**: 2026-05-04

## Source DGM-H reference

- **Repo**: devswha/dgmh
- **HEAD SHA**: 3eddf82a661688f88b4bbf4c55028509d0eac002
- **Date**: 2026-05-04
- **Plan reference**: dgmh/plans/hermes-skill-archive-dgmh-plan-v2-addendum.md (v2 supersedes v1 in §v1→v2 supersession table)

## Frozen reference files consulted for this port

| Reference file | Invariant established |
|---|---|
| playground/dgmh-engine/selectParents.ts | §8.2 sigmoid + novelty parent selection formula |
| playground/dgmh-engine/archive.ts | Append-only archive, archiveBumpChildren, archiveCap |
| playground/dgmh-engine/archiveRehydration.ts | Restart semantics, max(generationIndex)+1 |
| playground/dgmh-engine/types.ts | Generation, Archive, DgmhHyperparams field contracts |
| playground/dgmh-engine/loop.ts | Modifier→critic→eval→validator→admit ordering |
| playground/dgmh-engine/codexSubprocess.ts | Subprocess pattern: fail-closed, timeout, JSON envelope |
| playground/dgmh-engine/runLog.ts | Durable evidence shape obligations |
| playground/dgmh-engine/codexCritic.ts | Critic verdict JSON contract, fail-closed taxonomy |
| playground/dgmh-engine/selfModScope.ts | Mutation surface boundary principle |
| dgmh/critic/prompt-templates/modifier-prompt.md | Modifier role, JSON envelope, archive-summary framing |
| dgmh/critic/prompt-templates/modifier-critic.md | Critic role, approve/reject contract, fail-closed language |

## Archive schema version

`schemaVersion = 1` — used in archive.jsonl entries persisted to
`~/.hermes/dgmh/archive.jsonl`.

## Algorithm hyperparameters (paper §8.2 defaults)

- `lambda = 10`
- `topM = 3`
- `archiveCap = 50` (own-code, not in paper)
- `maxIterPerGeneration = 5` (own-code, not in paper)
