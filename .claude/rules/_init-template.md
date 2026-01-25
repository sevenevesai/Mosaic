# CLAUDE.md Initialization Template

**Purpose:** Use this template to create or update CLAUDE.md files for any project. This ensures consistent, effective AI assistance without requiring clarifying questions.

---

## When to Use

Run this initialization process at the start of any new session when:
- No CLAUDE.md exists
- CLAUDE.md is outdated (architecture changed, new systems added)
- Starting work on a new project
- Onboarding to an unfamiliar codebase

---

## Initialization Protocol

### Step 1: Codebase Analysis (Do First)

Gather these metrics before writing anything:

```
[ ] Lines of code by module (use cloc, tokei, or manual count)
[ ] File count and structure depth
[ ] Dependency count (package.json, Cargo.toml, requirements.txt)
[ ] Threading/async complexity (channels, mutexes, async/await patterns)
[ ] Test coverage (any tests? what kind?)
[ ] Build system (commands to build, test, run)
```

### Step 2: Identify Critical Information

Answer these questions - only include answers in CLAUDE.md if they're non-obvious:

**Architecture:**
- What are the 3-5 most important files/modules?
- What's the entry point?
- Are there worker threads, async tasks, or complex state machines?

**Patterns:**
- Any custom data formats or packed structures?
- Any non-standard coding patterns specific to this project?
- Any winding order, byte packing, or encoding details that are easy to get wrong?

**Gotchas:**
- What are the known issues/bugs?
- What are the common mistakes when working with this code?
- What will break silently if done wrong?

**Commands:**
- Exact build command (not "npm run build" if it's actually "npm run build:prod")
- Exact test command
- Any verification steps after changes

### Step 3: Write CLAUDE.md

Use this structure (target: 100-150 lines):

```markdown
# Project Name

One-line description with key metrics (LOC, complexity indicators).

## Quick Start
[Build and run commands - 5 lines max]

## Commands
[Table of exact commands - 10 lines max]

## Architecture
[ASCII tree of critical files only - 15 lines max]
[Skip obvious files like package.json, README.md]

## Key Constants
[Only if there are magic numbers/thresholds that affect behavior]

## Code Patterns
[Non-obvious patterns specific to this project - 20 lines max]
[Data formats, threading models, encoding details]

## Gotchas & Warnings
[Bullet list of things that will cause bugs if ignored - 10 lines max]

## Common Tasks
[Step-by-step for frequent operations - 20 lines max]

## Verification
[How to verify changes work - 10 lines max]

## Session Checklist
[4-5 bullet points for session start]
```

### Step 4: Create .claude/rules/ for Detailed Docs

If the project has complex subsystems, create separate files:

```
.claude/
└── rules/
    ├── _init-template.md    # This file (keep for future updates)
    ├── system-name.md       # Detailed docs for complex system
    └── another-system.md    # Another complex system
```

**Rule for splitting:** If explaining a system takes >30 lines and isn't needed for every task, put it in .claude/rules/.

---

## Quality Checklist

Before finalizing CLAUDE.md:

- [ ] Under 150 lines? (Ideally under 120)
- [ ] Every line answers: "Would removing this cause Claude to make mistakes?"
- [ ] No obvious information Claude already knows (standard language features, common patterns)
- [ ] No detailed API docs (link to docs instead)
- [ ] Commands are exact and copy-pasteable
- [ ] Gotchas are specific and actionable
- [ ] Architecture shows only critical files

---

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Full file tree | Bloats context | Show only critical 10-15 files |
| Tutorial-style explanations | Too verbose | Use bullet points, assume competence |
| Embedding entire docs via @ | Loads everything every time | Use "See path/to/doc.md" instead |
| "Don't do X" phrasing | Negative instructions are ignored | Use "Do Y instead of X" |
| Duplicating README content | Redundant | Reference README instead |
| Version history | Changes constantly | Keep in CHANGELOG |
| Code examples for standard patterns | Claude knows these | Only show project-specific patterns |

---

## Maintenance Protocol

Update CLAUDE.md when:
- [ ] Architecture changes (new modules, removed modules)
- [ ] New gotcha discovered (bug that was hard to find)
- [ ] Build/test commands change
- [ ] New common task pattern emerges

Include CLAUDE.md updates in the same commit as the change that necessitated them.

---

## Example: Minimal Effective CLAUDE.md

```markdown
# MyProject

Node.js REST API with PostgreSQL. ~8K LOC, 3 services, Redis caching.

## Quick Start
npm install && npm run dev

## Commands
| Action | Command |
|--------|---------|
| Dev server | `npm run dev` |
| Tests | `npm test -- --runInBand` |
| Lint | `npm run lint:fix` |
| DB migrate | `npm run db:migrate` |

## Architecture
src/
├── index.ts          # Entry, Express setup
├── routes/           # API endpoints
├── services/         # Business logic
├── models/           # Sequelize models
└── middleware/auth.ts # JWT validation (custom claims)

## Gotchas
- **Auth middleware** uses custom JWT claims; see middleware/auth.ts:45
- **Redis cache** has 5min TTL; invalidate manually on writes
- **DB transactions** must use `sequelize.transaction()`, not raw queries

## Verification
npm test && npm run lint
Check /health endpoint returns 200
```

This is 35 lines and contains everything needed for effective assistance.

---

## For Future Sessions

When starting a new session on this project:
1. Read CLAUDE.md (automatic)
2. Check if architecture has changed since CLAUDE.md was written
3. If yes, run initialization protocol to update
4. If no, proceed with task

The goal is zero clarifying questions for routine tasks.
