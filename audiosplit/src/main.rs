use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

/// AudioSplit command line interface.
#[derive(Debug, Parser)]
#[command(author, version, about = "Split audio files into tracks", long_about = None)]
struct Cli {
    /// Path to the input audio file.
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory where the split tracks will be written.
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    audiosplit_core::split_audio(&cli.input, &cli.output)
        .with_context(|| format!("failed to split '{}'", cli.input.display()))?;

    Ok(())
}
