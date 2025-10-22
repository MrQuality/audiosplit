mod cli;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context};
use audiosplit_core::Config;

use crate::cli::{build_cli, DEFAULT_POSTFIX};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let matches = build_cli().get_matches();

    let input_path = matches
        .get_one::<PathBuf>("file_path")
        .expect("required argument");
    if !input_path.is_file() {
        return Err(anyhow!(
            "input file does not exist: {}",
            input_path.display()
        ));
    }

    let segment_length = *matches
        .get_one::<Duration>("length")
        .expect("required argument");
    let output_dir = matches
        .get_one::<PathBuf>("output")
        .expect("defaulted argument");
    let postfix = matches
        .get_one::<String>("postfix")
        .cloned()
        .unwrap_or_else(|| DEFAULT_POSTFIX.to_owned());
    let overwrite = matches.get_flag("overwrite");

    let config = Config::builder(input_path, output_dir, segment_length, postfix)
        .overwrite(overwrite)
        .build()
        .with_context(|| {
            format!(
                "failed to create configuration for '{}'",
                input_path.display()
            )
        })?;

    audiosplit_core::run(config)
        .with_context(|| format!("failed to split '{}'", input_path.display()))?;

    Ok(())
}
