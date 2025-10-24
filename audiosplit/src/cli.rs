use audiosplit_core::{DEFAULT_BUFFER_FRAMES, DEFAULT_WRITE_BUFFER_SAMPLES, MAX_THREADS};
use clap::{builder::ValueParser, value_parser, Arg, ArgAction, Command};
use std::num::NonZeroUsize;
use std::path::PathBuf;

mod duration;

use duration::parse_duration;

const LENGTH_HELP: &str = concat!(
    "Length of each segment (e.g. 30s, 1m30s, 0.5h). ",
    "Supports +/- signs, fractional values, whitespace or underscores between components, ",
    "and the units ms, s, m, h, or d. Units may not repeat. Duration must be greater than zero."
);

pub const DEFAULT_POSTFIX: &str = "part";
const DEFAULT_BUFFER_FRAMES_STR: &str = "4096";
const DEFAULT_WRITE_BUFFER_SAMPLES_STR: &str = "8192";
const DEFAULT_THREADS_STR: &str = "1";

pub fn build_cli() -> Command {
    debug_assert_eq!(
        DEFAULT_BUFFER_FRAMES_STR
            .parse::<usize>()
            .expect("valid buffer frame default"),
        DEFAULT_BUFFER_FRAMES
    );
    debug_assert_eq!(
        DEFAULT_WRITE_BUFFER_SAMPLES_STR
            .parse::<usize>()
            .expect("valid write buffer default"),
        DEFAULT_WRITE_BUFFER_SAMPLES
    );
    debug_assert_eq!(
        DEFAULT_THREADS_STR
            .parse::<usize>()
            .expect("valid thread default"),
        1
    );

    Command::new(env!("CARGO_PKG_NAME"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("Split audio files into tracks")
        .version(env!("CARGO_PKG_VERSION"))
        .arg(
            Arg::new("length")
                .short('l')
                .long("length")
                .value_name("DURATION")
                .help(LENGTH_HELP)
                .required(true)
                .value_parser(ValueParser::new(parse_duration)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Directory where the split tracks will be written")
                .default_value(".")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("postfix")
                .short('p')
                .long("postfix")
                .value_name("POSTFIX")
                .help("Postfix inserted into generated file names")
                .default_value(DEFAULT_POSTFIX),
        )
        .arg(
            Arg::new("buffer-frames")
                .long("buffer-frames")
                .value_name("FRAMES")
                .help("Number of frames buffered in memory before flushing a segment")
                .value_parser(value_parser!(NonZeroUsize))
                .default_value(DEFAULT_BUFFER_FRAMES_STR),
        )
        .arg(
            Arg::new("write-buffer-samples")
                .long("write-buffer-samples")
                .value_name("SAMPLES")
                .help("Number of interleaved samples buffered before writing to disk")
                .value_parser(value_parser!(NonZeroUsize))
                .default_value(DEFAULT_WRITE_BUFFER_SAMPLES_STR),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("THREADS")
                .help("Number of worker threads to use when encoding segments")
                .default_value(DEFAULT_THREADS_STR)
                .value_parser(ValueParser::new(parse_thread_count)),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .help("Allow overwriting existing files in the output directory")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .help("Print the segment plan without writing any files")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("file_path")
                .value_name("FILE_PATH")
                .help("Path to the input audio file")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
}

fn parse_thread_count(arg: &str) -> Result<usize, String> {
    let value = arg
        .parse::<usize>()
        .map_err(|err| format!("invalid thread count: {err}"))?;

    if (1..=MAX_THREADS).contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "thread count must be between 1 and {} (received {value})",
            MAX_THREADS
        ))
    }
}
