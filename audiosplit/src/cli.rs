use clap::{builder::ValueParser, value_parser, Arg, ArgAction, Command};
use std::path::PathBuf;

mod duration;

use duration::parse_duration;

const LENGTH_HELP: &str = concat!(
    "Length of each segment (e.g. 30s, 1m30s, 0.5h). ",
    "Supports +/- signs, fractional values, whitespace or underscores between components, ",
    "and the units ms, s, m, h, or d. Units may not repeat. Duration must be greater than zero."
);

pub const DEFAULT_POSTFIX: &str = "part";

pub fn build_cli() -> Command {
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
            Arg::new("overwrite")
                .long("overwrite")
                .help("Allow overwriting existing files in the output directory")
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
