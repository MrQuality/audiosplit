mod cli;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Error};
use audiosplit_core::{AudioSplitError, Config};

use crate::cli::{build_cli, DEFAULT_POSTFIX};

fn main() {
    env_logger::init();

    let result: Result<(), anyhow::Error> = (|| {
        let matches = build_cli().get_matches();

        let input_path = matches
            .get_one::<PathBuf>("file_path")
            .expect("required argument");
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
        let dry_run = matches.get_flag("dry-run");

        let config = Config::builder(input_path, output_dir, segment_length, postfix)
            .create_output_dir(true)
            .overwrite(overwrite)
            .build()
            .with_context(|| {
                format!(
                    "failed to create configuration for '{}'",
                    input_path.display()
                )
            })?;

        if dry_run {
            let segments = audiosplit_core::dry_run(config).with_context(|| {
                format!("failed to simulate split for '{}'", input_path.display())
            })?;
            println!("Dry run - planned segments:");
            for path in segments {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| path.display().to_string());
                println!("  - {name}");
            }
        } else {
            match audiosplit_core::run(config) {
                Ok(()) => {}
                Err(err) => {
                    return Err(Error::new(err)
                        .context(format!("failed to split '{}'", input_path.display())));
                }
            }
        }

        Ok(())
    })();

    if let Err(error) = result {
        report_error(&error);
        std::process::exit(1);
    }
}

fn report_error(error: &anyhow::Error) {
    eprintln!("Error: {}", error);

    let mut causes = error.chain().skip(1).peekable();
    if causes.peek().is_some() {
        eprintln!("Caused by:");
        for cause in causes {
            eprintln!("  - {}", cause);
        }
    }

    eprintln!("\nTroubleshooting:");
    for tip in troubleshooting_tips(error) {
        eprintln!("  - {}", tip);
    }
}

fn troubleshooting_tips(error: &anyhow::Error) -> &'static [&'static str] {
    for cause in error.chain() {
        if let Some(split_error) = cause.downcast_ref::<AudioSplitError>() {
            return match split_error {
                AudioSplitError::InvalidPath(_) => &[
                    "Verify the input file exists and that the path is spelled correctly.",
                    "If the path is relative, try providing an absolute path to the file.",
                ],
                AudioSplitError::OutputExists(_) => &[
                    "Remove the existing segments from the output directory before re-running.",
                    "Re-run with --overwrite to replace existing segments in place.",
                ],
                AudioSplitError::InvalidSegmentLength { .. } => &[
                    "Choose a segment length greater than zero.",
                    "Use time units such as '500ms' or '2m' when specifying --length.",
                ],
                AudioSplitError::MissingOutputDirectory(_) => &[
                    "Ensure the output directory exists or allow the tool to create it.",
                    "Double-check the --output argument for typos or missing folders.",
                ],
                AudioSplitError::OutputDirectoryNotWritable(_) => &[
                    "Confirm you have permission to write to the selected output directory.",
                    "Pick a different output location with write access or adjust permissions.",
                ],
                AudioSplitError::InsufficientDiskSpace { .. } => &[
                    "Free up disk space in the output location before splitting.",
                    "Reduce the segment length to produce smaller output files.",
                ],
                _ => DEFAULT_TROUBLESHOOTING_TIPS,
            };
        } else if cause.is::<std::io::Error>() {
            return &[
                "Verify the input file can be read and the output directory is writable.",
                "Check whether another process is using the files and close it before retrying.",
            ];
        }
    }

    DEFAULT_TROUBLESHOOTING_TIPS
}

const DEFAULT_TROUBLESHOOTING_TIPS: &[&str] = &[
    "Verify the input and output paths are correct.",
    "Ensure you have permission to access the input file and write to the output directory.",
    "Re-run with --dry-run to inspect the planned segments without writing files.",
];
