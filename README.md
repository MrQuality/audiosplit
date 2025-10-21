# AudioSplit

AudioSplit is a command-line tool for splitting long audio recordings into uniform segments. It wraps the `audiosplit_core` library and exposes a friendly interface for choosing segment length, destination directory, and a postfix used in the generated filenames.

## Features

- Split any supported audio format into equal-length chunks using Symphonia decoders.
- Configure the segment duration in milliseconds for precise control over output length.
- Choose the destination directory where resulting files are written.
- Customize the filename postfix added between the base name and the segment counter.
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

Usage: audiosplit [OPTIONS] --length <MILLISECONDS> --output <OUTPUT_DIR> <FILE_PATH>

Arguments:
  <FILE_PATH>  Path to the input audio file

Options:
  -l, --length <MILLISECONDS>  Length of each segment in milliseconds
  -o, --output <OUTPUT_DIR>    Directory where the split tracks will be written
  -p, --postfix <POSTFIX>      Postfix inserted into generated file names [default: part]
  -h, --help                   Print help
  -V, --version                Print version
```

### Example

Split a recording into 30-second parts and save them to the `segments/` directory with a custom postfix:

```bash
audiosplit --length 30000 --output segments --postfix chapter meeting.mp3
```

The command above will create output files named similar to `meeting.chapter.001.mp3` inside the `segments/` directory.

## License

AudioSplit is distributed under the terms of the [MIT License](LICENSE-MIT).
