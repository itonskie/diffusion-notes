---
type: meta
tags:
  - meta
  - log
date: 2026-06-29
---

# Wiki Logs

Chronological record of wiki activity.

## 2026-06-29

### Wiki Initialization
- Scaffolded wiki structure on top of existing diffusion-notes repo
- Created schema, index, source index, kanban, logs
- Categories: `concepts/`, `papers/`, `experiments/`, `workflows/`, `configs/`, `reference/`
- Migrated `theory/00-index.md` reading list into [[foundations]] MOC
- Removed legacy `theory/` and `lab-log/` directories (content carried into MOC / experiments guidance)
- Set up Obsidian Bases for dynamic views (all-pages, recent, kanban, by-topic)

### Ingested: HF Diffusion Course — Unit 1 intro
- Source: https://huggingface.co/learn/diffusion-course/unit1/1
- Pages created: [[diffusion-models]], [[generative-models]], [[ddpm]] (stub, status inbox), [[reference/diffusion-course]]
- Pages updated: [[foundations]] (wikilinks now resolve to created pages)
- Notebooks not yet worked through — captured as reference in [[reference/diffusion-course]]

### Ingested: Lilian Weng — "What are Diffusion Models?"
- Source: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Pages created: [[scheduler]], [[classifier-free-guidance]], [[u-net]], [[dit]], [[latent-diffusion]]
- Pages updated: [[ddpm]] (status inbox → exploring; added forward/reverse math, simplified loss, sampling algorithm, score-matching connection); [[diffusion-models]] (added score-matching section, wikilinks to new pages)
- Skipped pre-stubbing: Lilian's post mentions DDIM, Improved DDPM, LDM, ControlNet, Imagen, Consistency Models — these get wikilinks (unresolved in Obsidian) but no stub files until they're actually targeted for ingest

### Ingested: HF Diffusion Course — Unit 2 intro
- Source: https://huggingface.co/learn/diffusion-course/unit2/1
- Pages created: [[fine-tuning]], [[guidance]]
- Pages updated: [[u-net]] (added "Three ways to inject conditioning" section); [[reference/diffusion-course]] (added Unit 2 section + table status)
- Notebooks not yet worked through — captured as reference

### Ingested: DDPM paper
- Source: https://arxiv.org/abs/2006.11239 (abstract page + ar5iv HTML render)
- Pages updated: [[ddpm]] (status exploring → developing; added abstract verbatim, ablation Table 2, headline FID/IS results, U-Net details from Appendix B refs, progressive lossy decompression interpretation, score-matching equivalence)
- PDF fetch returned binary — used ar5iv HTML render at https://ar5iv.labs.arxiv.org/html/2006.11239 to get past the abstract. Still missing from page: full optimizer config, EMA decay, batch size — would need to read Appendix B directly. Logged as open in the page.

### Ingested: LoRA paper
- Source: https://arxiv.org/abs/2106.09685 (via ar5iv HTML)
- Pages created: [[papers/lora]] (paper-specific page with abstract, rank ablation Table 6, layer-selection Table 5, parameter savings, NLP results); [[lora]] (concept page focused on diffusion application: layer selection per-tool, choosing r for diffusion, alpha/strength, caption strategy, sprint workflow); [[rank-decomposition]] (math intuition: SVD, Eckart-Young, parameter count savings, why init matters)
- ar5iv HTML render was rich — got all needed Table 2/5/6 ablations, hyperparams, parameter counts
