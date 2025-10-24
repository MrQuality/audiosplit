# AudioSplit

AudioSplit is a command-line tool for splitting long audio recordings into uniform segments. It wraps the `audiosplit_core` library and exposes a friendly interface for choosing segment length, destination directory, and a postfix used in the generated filenames.

## Features

- Split any supported audio format into equal-length chunks using Symphonia decoders.
- Configure the segment duration with readable strings such as `30s`, `1h30m`, or `0.5d` for precise control over output length.
- Choose the destination directory where resulting files are written.
- Customize the filename postfix added between the base name and the segment counter.
- Control the number of worker threads used for encoding to match your system resources.
- Receive clear error messages when the input file cannot be processed.

## Installation

You will need the Rust toolchain (edition 2021) installed. The recommended way to install the CLI is via `cargo`:

```bash
cargo install --path audiosplit
```

Alternatively, you can build and run the binary directly from the repository:

```bash
cargo run --package audiosplit -- <options>
```

## Usage

The CLI help output is reproduced below and is kept in sync with the latest implementation:

```
Split audio files into tracks

Usage: audiosplit [OPTIONS] --length <DURATION> --output <OUTPUT_DIR> <FILE_PATH>

Arguments:
  <FILE_PATH>  Path to the input audio file

Options:
  -l, --length <DURATION>
          Length of each segment (e.g. 30s, 1m30s, 0.5h). Supports +/- signs,
          fractional values, whitespace or underscores between components, and
          the units ms, s, m, h, or d. Units may not repeat. Duration must be
          greater than zero.
  -o, --output <OUTPUT_DIR>
          Directory where the split tracks will be written
  -p, --postfix <POSTFIX>
          Postfix inserted into generated file names [default: part]
      --threads <THREADS>
          Number of worker threads to use when encoding segments [default: 1] [possible
          values: 1..=32]
  -h, --help
          Print help
  -V, --version
          Print version
```

### Example

Split a recording into 30-second parts and save them to the `segments/` directory with a custom postfix:

```bash
audiosplit --length 30s --output segments --postfix chapter meeting.mp3
```

The command above will create output files named similar to `meeting.chapter.001.mp3` inside the `segments/` directory.

### Duration format

Duration strings accepted by `--length` follow the grammar below:

```text
duration   = [ sign ] separators? component ( separators component )* ;
sign       = "+" | "-" ;
component  = number separators? unit ;
number     = digits [ "." digits ] ;
digits     = digit , { digit } ;
unit       = "ms" | "s" | "m" | "h" | "d" ;
separators = { whitespace | "_" } ;
```

- Units are case-sensitive and may appear at most once; duplicates are rejected to
  avoid precedence ambiguities.
- Fractional values are supported up to the precision implied by the unit (for
  example milliseconds accept up to six decimal places).
- Leading `+` signs are allowed, but negative or zero durations are rejected.
- The total duration must not exceed `u64::MAX` milliseconds.

## License

AudioSplit is distributed under the terms of the [MIT License](LICENSE-MIT).
