# Source Architecture

This repository is transitioning toward a production-oriented src-based package structure.

## Planned Structure

```text
src/
├── filters/
├── realtime/
├── evaluation/
├── visualization/
└── utils/
```

## Goals

- modular DSP architecture
- separation of runtime and evaluation logic
- maintainable adaptive filtering workflows
- production-style engineering organization
- easier experimentation and benchmarking

## Migration Plan

Current modules under `model/` will progressively migrate into:

- `src/filters`
- `src/realtime`
- `src/evaluation`
- `src/visualization`

as the repository evolves toward a cleaner package layout.
