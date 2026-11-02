# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Fixed

- Post a failed turn's reply as a Linear `error` activity (recognised by the gateway's trailing `FAILED_TURN_NOTICE` / `PARTIAL_FAILED_TURN_NOTICE`) so the agent session shows `error` instead of a spurious `complete`. Normal replies still post `response`; kanban pings still post `thought`.
- The `linear_agent_*` tools now work in processes where the gateway adapter never connected (desktop chat, CLI, cron). With no active adapter, the registry lazily builds one fallback adapter from the effective platform config — no webhook listener, no platform lock, no viewer query — reusing the cached OAuth token and only minting a new one when it expires. A connected gateway adapter still takes precedence.

## 0.4.2 — 2026-07-29

### Fixed

- Resolve a child `AgentSession.sourceCommentId` to its root comment before replying; Linear rejects child comments as `commentCreate.parentId` with `incorrect parent`.
- Mirror final Agent Session responses to the resolved source thread in adapter-controlled delivery instead of relying on the model to invoke a comment tool.

## 0.4.1 — 2026-07-29

### Fixed

- Preserve Linear's current `agentSession.sourceCommentId` webhook field so replies triggered from an existing issue-comment thread can be routed back to that original thread.
- Accept `agentActivity.sourceCommentId` as a compatible fallback for prompted activity payloads.

## 0.4.0 — 2026-07-29

### Added

- Standalone Hermes platform-plugin distribution for Linear Agent Sessions.
- Git-directory plugin manifest and pip entry point.
- 58 attributed `linear_agent_*` tools with fail-closed mutation policies.
- Linear client-credentials and authorization-code OAuth support.
- Signed webhook handling, replay protection, Agent Activities, plans, session links, clarification, stop signals, and standalone cron delivery.
- Plugin-owned, process-locked, atomic OAuth state storage with POSIX mode `0600`.
- Read-only migration fallback for earlier `providers.linear_agent` state in Hermes `auth.json`.
- Compatibility tests for current Hermes upstream and Python 3.11–3.13.
