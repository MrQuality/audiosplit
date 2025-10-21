use std::path::PathBuf;

use anyhow::{anyhow, Context};
use audiosplit_core::Config;
use clap::{value_parser, Arg, Command};

const DEFAULT_POSTFIX: &str = "part";

fn cli() -> Command {
    Command::new(env!("CARGO_PKG_NAME"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("Split audio files into tracks")
        .version(env!("CARGO_PKG_VERSION"))
        .arg(
            Arg::new("length")
                .short('l')
                .long("length")
                .value_name("MILLISECONDS")
                .help("Length of each segment in milliseconds")
                .required(true)
                .value_parser(value_parser!(u64).range(1..)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Directory where the split tracks will be written")
                .required(true)
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
            Arg::new("file_path")
                .value_name("FILE_PATH")
                .help("Path to the input audio file")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let matches = cli().get_matches();

    let input_path = matches
        .get_one::<PathBuf>("file_path")
        .expect("required argument");
    if !input_path.is_file() {
        return Err(anyhow!(
            "input file does not exist: {}",
            input_path.display()
        ));
    }

    let length_ms = *matches.get_one::<u64>("length").expect("required argument");
    let output_dir = matches
        .get_one::<PathBuf>("output")
        .expect("required argument");
    let postfix = matches
        .get_one::<String>("postfix")
        .cloned()
        .unwrap_or_else(|| DEFAULT_POSTFIX.to_owned());

    let config = Config::new(input_path, output_dir, length_ms, postfix).with_context(|| {
        format!(
            "failed to create configuration for '{}'",
            input_path.display()
        )
    })?;

    audiosplit_core::run(config)
        .with_context(|| format!("failed to split '{}'", input_path.display()))?;

    Ok(())
}
