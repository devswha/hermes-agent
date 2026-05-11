# dgmh-webhook-test

Iterative Discord webhook validation for the DGM-H flask persona.

Sends N test prompts via a channel webhook, waits a configurable
interval between each, then parses `~/.hermes/logs/agent.log` to
extract per-iteration metrics (response length, patina rewrite,
gate decision, api_calls). Outputs a markdown comparison table.

## Why

Operator-driven validation of SOUL.md / patina / pre-send-gate /
soul-evolution changes without manually typing each test message
into Discord. Same path the live bot uses (webhook → `on_message`
→ agent → adapter.send), so the response is a true end-to-end test.

## Usage

```bash
# minimum: prompts from CLI, webhook URL from env file
python3 ~/.hermes/skills/dgmh-webhook-test/bin/dgmh-webhook-test.py \
  --webhook-env ~/workspace/dgmh/.env.local \
  --prompts "넌 누구야" "프로젝트 깨끗해?" "오늘 비트코인 가격 알아?"

# from a prompt file (one per line, blank lines and `#` comments ignored)
python3 ~/.hermes/skills/dgmh-webhook-test/bin/dgmh-webhook-test.py \
  --webhook-env ~/workspace/dgmh/.env.local \
  --prompts-file /tmp/iter-prompts.txt \
  --interval-s 35 \
  --output /tmp/iter-result.json
```

## Arguments

- `--webhook URL` — Discord webhook URL. OR
- `--webhook-env PATH` — env file with a `webhook=...` line (e.g. `.env.local`).
- `--prompts STR [STR ...]` — inline prompts.
- `--prompts-file PATH` — one prompt per line (`#` comments + blank lines stripped).
- `--interval-s FLOAT` — seconds between sends. Default 35 (≥ bimodal sleep ceiling so each turn closes before the next).
- `--log-path PATH` — agent log. Default `~/.hermes/logs/agent.log`.
- `--output PATH` — optional JSON dump of per-iteration data.
- `--quiet` — suppress per-send progress lines.

## Exit codes

- 0 — all sends succeeded
- 1 — webhook URL missing or invalid
- 2 — one or more webhook POSTs returned non-2xx
- 3 — log scan failed (file unreadable)

## Notes

- The webhook account must be in `DGMH_ALLOWED_BOT_USERS` on the
  systemd drop-in, otherwise the gateway's bot-filter drops the
  inbound and the bot never replies.
- Log parsing relies on the inbound message text appearing verbatim
  in `gateway.run: inbound message`. Truncated inbounds (Discord
  shortens long messages in some clients) may miss the match.
- DGM-H mission alignment: validates `ai_score`, `human_likeness`,
  truncation rate, and "잘 몰라" rate over a batch.
