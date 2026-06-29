---
type: meta
tags:
  - meta
  - schema
date: 2026-06-29
---

# Wiki Schema

Conventions and structure for this wiki.

## Purpose

Notes, theory, and experiments from a 6-week diffusion fine-tuning sprint — diffusion models, LoRA training, ComfyUI workflows, and the engineering practices around them.

## Page Types

### Maps of Content (`mocs/`)
Entry points into a topic area. Like a Wikipedia article page — provides narrative context, explains how concepts relate, and tells you what to read and in what order. Create a MOC when a topic has 3+ related pages that benefit from a guided overview.
- **Filename**: `mocs/<topic-slug>.md`
- **Structure**: Introduction, narrative walkthrough linking to pages in a logical reading order, embedded base query, See Also

### Concepts (`concepts/`)
Theory and architectural ideas — denoising, schedulers, classifier-free guidance, low-rank adaptation, rank decomposition. One concept per page. Concepts are the "what is X" entries.
- **Filename**: `concepts/<slug>.md`
- **Structure**: Definition, intuition, math sketch (only if it earns its place), why it matters in practice, See Also

### Papers (`papers/`)
One page per paper. Captures the core idea, the contribution, what to remember, and which concepts it grounds.
- **Filename**: `papers/<short-name>.md` (e.g. `papers/ddpm.md`, `papers/lora.md`)
- **Structure**: Citation, problem, key contribution, intuition, what's not in the paper, related concepts, See Also

### Experiments (`experiments/`)
Lab log — one entry per training run or notable experiment. What you tried, what config, what happened, cost, what you'd do differently.
- **Filename**: `experiments/YYYY-MM-DD-<short-slug>.md`
- **Structure**: Goal, setup, command/config, result, cost, lessons, See Also

### Workflows (`workflows/`)
ComfyUI node graphs and recipes. Screenshots, JSON exports, short notes per workflow.
- **Filename**: `workflows/<slug>.md`
- **Structure**: Purpose, screenshot, node walkthrough, gotchas, See Also

### Configs (`configs/`)
AI-Toolkit / Musubi Tuner / other training configs that completed a run. Annotated with comments on each meaningful setting.
- **Filename**: `configs/<trainer>-<model>-<descriptor>.yaml` for the config itself, with a sibling `.md` if extended notes are needed
- **Structure** (for notes file): Trainer + model, dataset, what was trained, results link, See Also

### Reference (`reference/`)
Cheatsheets, comparison tables, model cards, "which scheduler when" — the lookup material you don't want to derive every time.
- **Filename**: `reference/<slug>.md`
- **Structure**: Variable, but use tables liberally

## Conventions

### Frontmatter
Every wiki page MUST have YAML frontmatter:
```yaml
---
type: <page-type>           # concept | paper | experiment | workflow | config | reference | moc
tags:
  - <relevant-tags>         # lowercase, hyphenated
date: YYYY-MM-DD            # date created
updated: YYYY-MM-DD         # date last modified
status: <status>            # optional — inbox | exploring | summary | done
---
```

#### Status property
The `status` field is optional and tracks how deeply you've engaged with a source — most useful on paper pages and experiments.
- **inbox** — captured but not yet explored
- **exploring** — actively reading/researching
- **summary** — main contributions captured on the wiki page, original source not yet deep-read end-to-end
- **done** — fully deep-read and understood (papers) / completed and analyzed (experiments)

Use `status` on **papers and experiments**. Concepts, references, and MOCs don't need it — they're working summaries that get iteratively improved, not items with a "fully read" finish line.

### Linking
- Use **Obsidian wikilinks**: `[[page-name]]` or `[[page-name|display text]]`
- **Never use display-text wikilinks inside markdown tables** — the `|` in `[[page|text]]` breaks the table. Use `[[page-name]]` without display text in tables
- Do NOT use markdown links for internal wiki pages
- Use wikilinks **inline throughout the body text** wherever a concept is naturally mentioned — don't save all links for the footer
- Every page should have a `## See Also` section at the bottom for related pages that don't come up naturally in the body
- When creating or updating a page, check existing pages for places that should link back to the new content

### Tags
- Use YAML `tags:` in frontmatter (not inline `#tags` in body text)
- Keep tags lowercase and hyphenated: `lora`, `cfg`, `flux`, `scheduler`, `ddpm`
- Tags complement the folder structure — a page lives in one folder but can have many tags

### Canvas (`canvas/`)
Freeform visual thinking spaces for brainstorming or mapping relationships before structuring into pages.
- **Filename**: `canvas/<slug>.canvas`
- Canvas files are JSON, edited in Obsidian's visual editor
- Not managed by Claude — these are your spatial thinking tool

### Images and Assets
- Store images in `assets/`
- Embed with Obsidian syntax: `![[filename.png]]`
- Download images locally rather than linking externally
- For ComfyUI screenshots, prefer PNG with the workflow embedded so it can be re-imported

### Logs
- `logs.md` tracks wiki activity chronologically
- Group by date
- Log what was researched, pages created/updated, sources used

### Bases (`bases/`)
Obsidian Bases provide dynamic, auto-updating views of wiki content.

- **`bases/all-pages.base`** — all wiki pages grouped by type
- **`bases/recent.base`** — recently updated pages (limit 20)
- **`bases/kanban.base`** — pages with a `status` property, grouped by status
- **`bases/by-topic.base`** — all pages grouped by tag

Embed with `![[bases/filename.base]]` or inline with a ` ```base ` code block.

#### Bases in MOCs
Each MOC should include an embedded base query at the bottom (before See Also) that auto-pulls all pages sharing its primary tag. This keeps new pages discoverable even if not yet linked in the narrative.

```base
filters:
  and:
    - file.hasTag("topic-tag")
    - 'type != "moc"'
views:
  - type: table
    name: Pages
    order:
      - file.name
      - type
      - updated
```

### Index
- `index.md` is the central narrative map — curated MOC links and a changelog
- The exhaustive page listing is handled by `bases/all-pages.base`, embedded in index

### Sources
- `sources/index.md` logs every ingested source
- Each entry: date, URL/title, which wiki pages it informed
