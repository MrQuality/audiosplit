# Contributing to AudioSplit

Thanks for your interest in contributing! The following guidelines help keep the project healthy and consistent.

## Development Environment

1. Install the latest stable Rust toolchain using [`rustup`](https://rustup.rs/).
2. Clone this repository and fetch dependencies:
   ```bash
   git clone https://github.com/<your-org>/audiosplit.git
   cd audiosplit
   cargo fetch
   ```
3. (Optional) Enable automatic formatting on save in your editor using `rustfmt`.

## Coding Standards

- Follow idiomatic Rust style and keep functions focused and well-documented when behavior is non-obvious.
- Always format Rust code with `cargo fmt` before committing.
- Run `cargo clippy --all-targets --all-features` and address warnings or provide clear justification in the commit message if a warning cannot be fixed.
- Prefer small, cohesive commits with descriptive messages.
- Include tests for new functionality or bug fixes whenever possible. Integration tests live under `audiosplit/tests/` and core library tests in `audiosplit_core/`.

## Continuous Integration Expectations

Our CI pipelines run the following commands. Make sure they pass locally before pushing changes:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test --all --all-features
```

Pull requests that fail these checks cannot be merged. If you introduce new tooling or scripts, document them in this file.

## Benchmarks

Benchmarks live under `audiosplit_core/benches` and use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) for measuring performance. The `split_benchmark` target synthesizes short WAV fixtures on the fly, so no binary assets need to be checked in.

Run all benchmarks with:

```bash
cargo bench -p audiosplit_core
```

You can also focus on the split benchmark specifically:

```bash
cargo bench -p audiosplit_core split_benchmark
```

Ensure you have the Rust toolchain installed; Criterion will handle warmup and sampling automatically.

## Reporting Issues

If you encounter a bug or have a feature request, open an issue with clear reproduction steps and expected behavior. Screenshots or log excerpts are helpful when available.

## Code of Conduct

We expect all contributors to interact with respect and professionalism. Harassment of any kind will not be tolerated.
