# Continuum

A real-time renderer built on an original paradigm: certified incremental cone
rendering. Every visibility question is asked by a cone that answers in bits,
and cached answers are recomputed only when they could have changed by more
than a chosen, invisible error bound.

Status: research stage. The core bets are being tested in 2D before any GPU work.
See [docs/DESIGN.md](docs/DESIGN.md) for the design and
[docs/results/](docs/results/) for measured results.

## Layout

- `crates/continuum`: the library (currently the direction-mask primitive).
- `experiments/flatland`: 2D experiments that test the core bets.
- `docs/`: design document and experiment results.

## Running

```bash
cargo test --release
cargo run --release -p flatland
```

The flatland run prints a results table and writes images to
`experiments/flatland/out/` (`sheet.png` is the overview: top row renders,
bottom row signed error maps, red too dark, blue too bright).
