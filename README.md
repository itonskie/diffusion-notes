# Diffusion Notes

> **This repo is an Obsidian vault.** It uses `[[wikilinks]]` and Obsidian Bases (`.base` files) for interconnected note-taking. GitHub renders the markdown files fine, but wikilinks won't be clickable and `.base` views won't render. For the full experience: clone, then open the project root as a vault in [Obsidian](https://obsidian.md). Otherwise, start at [`index.md`](./index.md) and follow the links.

Open study notes and lab logs from my [diffusion engineering sprint](https://github.com/itonskie/diffusion-sprint). Coming from an LLM-and-agents AI engineering background, learning image/video diffusion in public.

Not polished. Not authoritative. These are working notes — what surprised me, what tripped me up, what I had to look up twice.

## Where to start

- **[`index.md`](./index.md)** — central map of all pages.
- **[`mocs/foundations.md`](./mocs/foundations.md)** — Week 1 reading path (HF Diffusion Course → key papers → architecture-specific reads).
- **[`schema.md`](./schema.md)** — wiki conventions if you care about how it's structured.

## Vault structure

| Dir | What's in it |
|-----|--------------|
| `mocs/` | Maps of Content — narrative entry points into a topic area |
| `concepts/` | Theory pages — one concept per page (denoising, schedulers, CFG, rank decomposition) |
| `papers/` | One page per paper (DDPM, LoRA, DiT, etc.) — citation + intuition + key contribution |
| `experiments/` | Lab log — one entry per training run, `YYYY-MM-DD-<slug>.md` |
| `workflows/` | ComfyUI node graphs — screenshots, JSON exports, notes |
| `configs/` | AI-Toolkit / Musubi Tuner configs that completed a run, annotated |
| `reference/` | Cheatsheets, comparison tables, model cards |
| `sources/` | Log of every ingested source and which pages it informed |
| `bases/` | Obsidian Bases — dynamic table views over the vault |
| `canvas/` | Obsidian Canvas files for visual brainstorming |
| `assets/` | Images, screenshots |

## Sprint context

See [diffusion-sprint](https://github.com/itonskie/diffusion-sprint) for the full 6-week plan and progress log. This repo is the public knowledge artifact built up alongside the sprint.
