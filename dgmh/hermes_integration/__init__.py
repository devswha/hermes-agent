"""
dgmh/hermes_integration — W6: Live reaction-triggered SOUL.md evolution.

Hook surface investigation findings (2026-05-04):
  - gateway/hooks.py exposes: gateway:startup, session:start/end/reset,
    agent:start/step/end, command:* — NO discord:reaction event.
  - Discord client (gateway/platforms/discord.py) registers on_message and
    on_voice_state_update via @client.event, but no on_raw_reaction_add.
  - Discord intents: message_content, dm_messages, guild_messages, voice_states.
    Reactions require the 'reactions' intent (part of Intents.default()).
    Intents.default() IS used, so guild message reactions are received.
  - Approach: gateway:startup hook that locates the DiscordAdapter instance
    and monkey-patches on_raw_reaction_add onto its _client after on_ready.
  - run_agent.py::_spawn_background_review (line 2879) needs a gate check
    reading dgmh.gate_native_review from ~/.hermes/config.yaml.
"""
