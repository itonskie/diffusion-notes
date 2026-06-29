# CLAUDE.md — diffusion-notes

This repo is an LLM Wiki (Karpathy pattern). The project root is the Obsidian vault. It's the public learning record for Mark's 6-week [diffusion sprint](https://github.com/itonskie/diffusion-sprint).

## Start of every session
1. Read `index.md` for the full map of existing knowledge
2. Read `schema.md` to understand wiki conventions (page types, frontmatter, linking rules)
3. Build on existing knowledge — don't duplicate pages

## Adding knowledge to the wiki

- **User provides a URL/file** → use the `/ingest` skill to process it into wiki pages.
- **User asks to learn about a topic** ("learn about X", "look into Y", "research Z") → search the web for the best documentation, then follow the full `/ingest` workflow: create/update wiki pages with proper frontmatter and `[[wikilinks]]`, log the source in `sources/index.md`, update `logs.md`, and update `index.md`. Treat web research the same as a URL ingest.
- **User points to a local file or directory** → read the content and follow the same `/ingest` workflow.

## Maps of Content (MOCs)

- MOCs are entry points into a topic area — like a Wikipedia article page. Narrative context + reading order + linked pages.
- Create a MOC when a topic has 3+ related pages that benefit from a guided overview.
- When ingesting content, check if a relevant MOC exists and update it. If a growing topic has no MOC yet, create one.
- Each MOC should include an embedded base query (```base code block) at the bottom that auto-pulls all pages sharing its primary tag.

## Linking

- Use `[[wikilinks]]` **inline throughout the body text** wherever a concept is naturally mentioned, not just in the `## See Also` footer.
- **Never use display-text wikilinks inside markdown tables** — the `|` in `[[page|text]]` breaks the table. Use `[[page-name]]` without display text in tables.
- The `## See Also` section is for related pages that don't come up naturally in the body — discovery links.
- When creating or updating a page, also check existing pages for places that should link back to the new content.

## Tone for new pages

- Plain English, short sentences. Mark is a senior engineer — no basics he already knows.
- Notes are working notes, not polished tutorials. "What surprised me, what tripped me up, what I had to look up twice."
- For concept and paper pages: intuition first, math sketch only if it earns its place.

## Working agreements

- Fully public repo on GitHub `itonskie`.
- Git identity for this repo: `Mark Oesterdal <40949991+itonskie@users.noreply.github.com>` (per-repo, not global).
- Don't update global git config.
- `.safetensors`, `.ckpt`, `.pt`, `.bin` are already gitignored — don't commit model artifacts.
