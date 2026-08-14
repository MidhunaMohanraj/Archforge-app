# ArchForge - prototype

A working starting point for the ArchForge architecture: an on-premise
assistant that drafts an answer, grounds it in your own reference
documents, and validates it against your coding standards before an
engineer ever sees it.

This is a prototype of the **intelligence pipeline** at the center of the
framework (deck slide 5, "How each answer is produced"). It is not the
full platform - there's no GUI studio, fine-tuning trainer, or multi-user
serving layer yet. Those are separate, larger pieces of work; this gives
us something real to test the core idea against tonight, and a clean
base to build the rest on.

## What it does

Every query goes through three stages:

1. **Draft** - the configured model writes a first answer in a system
   prompt that asks for the team's own conventions.
2. **Ground** - the query is matched against your reference documents
   (`sample_docs/` for now) using a TF-IDF search, and the model is
   asked to correct any specifics against what it finds. The answer
   comes back tagged with which document it was grounded on.
3. **Validate** - the answer is checked against a small subset of
   MISRA C:2012 (no `goto`, no dynamic memory allocation, braces
   required on compound statements, switch/case fallthrough). If it
   fails, the model gets the specific violations and one to two chances
   to fix them before it's returned.

## Running it

Requires Python 3.10+, no third-party packages.

```bash
# Point it at your model server (defaults to http://localhost:8000/v1,
# which matches a local vLLM or Ollama OpenAI-compatible endpoint)
python -m archforge.cli init-config
# edit archforge.config.json if your endpoint or model name differs

python -m archforge.cli ask "Write a CAN driver init function for the M_CAN peripheral"
```

Add `--log` to save the full draft / ground / validate trace under
`logs/` - useful for reviewing what the model actually did at each
stage, not just the final answer.

If no model endpoint is reachable, the CLI still runs end to end using a
clearly-labelled offline stub in place of a real model response, so the
retrieval and validation stages can be demoed without a live server.

## Project layout

```
archforge/
  config.py      site configuration: model endpoint, docs dir, ruleset
  llm_client.py  the one place that talks to the model endpoint
  retrieval.py   TF-IDF grounding over sample_docs/
  validator.py   standards checker + self-repair loop
  pipeline.py    wires draft -> ground -> validate together
  cli.py         command-line entry point
sample_docs/     example reference doc used for grounding
```

## Extending this toward the full framework

Each stage was written behind a narrow interface on purpose, so the
pieces described in the deck can slot in without reworking the pipeline:

- **Retrieval** - swap `RetrievalIndex` for an embeddings-based vector
  index once you have a real corpus; `pipeline.py` only calls `.search()`.
- **Validation** - swap `StandardsChecker` for a real MISRA tool (e.g.
  cppcheck with a MISRA addon); `SelfRepairValidator` only calls `.check()`.
- **Serving multiple engineers** - wrap `ArchForgePipeline.run()` behind a
  small API server (FastAPI is a natural fit) so it can be called from an
  IDE plugin instead of the CLI.
- **Fine-tuning / GUI studio** - separate from this pipeline entirely;
  this repo assumes a model is already being served at `model.base_url`.
