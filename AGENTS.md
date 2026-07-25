# AGENTS.md

<!-- codex-kit-managed-start -->
## Codex Kit

This project has a local Codex kit installed.

- Project signals detected by installer: documents
- Skills are available under `.agents/skills`; use the most relevant skill before specialized work.
- Skill support files such as install scripts and shared helpers are available under `.agents/skills`.
- Custom agents are configured under `.codex/agents`; spawn subagents only when the task benefits from delegation.
- Hooks may enforce privacy, scout checks, and post-edit simplification reminders.
- Workflow rules are available under `.codex/rules`; read relevant rules before planning or implementation.
- Helper scripts are available under `.codex/scripts`; use these project-local paths for kit utilities.
- Active plan state, when set, is stored at `.codex/state/active-plan.json`; check it before implementing plan-driven work.
- Plan templates are available under `plans/templates`; create plans/reports in this kit target directory unless the user explicitly asks for a child repo plan.
- CodexKit reference docs are available under `.codex/docs`; use them for agent-team guidance, code standards, architecture, skill maps, and research notes.
- Output style references are available under `.codex/output-styles`; use them as tone/detail guides when the user asks for a specific level.
- Keep changes scoped to the user's request and preserve existing project conventions.

Installed agents: brainstormer, planner, researcher, code_reviewer, tester, debugger, git_manager, docs_manager

Installed skills: ask, brainstorm, plan, cook, research, scout, code-review, test, debug, fix, git, docs, problem-solving, context-engineering, docx, pdf, pptx, xlsx

Enabled hooks: session-init, subagent-init, dev-rules-reminder, privacy-block, scout-block, cook-after-plan-reminder, post-edit-simplify-reminder, descriptive-name
<!-- codex-kit-managed-end -->
