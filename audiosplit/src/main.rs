mod cli;

use std::convert::TryInto;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context};
use audiosplit_core::{Config, ProgressReporter};
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};

use crate::cli::{build_cli, DEFAULT_POSTFIX};

struct CliProgress {
    bar: ProgressBar,
    total: Option<u64>,
    finished: bool,
}

impl CliProgress {
    fn new(input: &Path) -> Self {
        let bar = ProgressBar::new_spinner();
        let style = ProgressStyle::with_template(
            "{spinner:.green} {msg}{wide_bar:.cyan/blue} {pos:>7}/{len:>7} ({eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar());
        let mut progress = Self {
            bar,
            total: None,
            finished: false,
        };
        let name = input
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_owned())
            .unwrap_or_else(|| input.to_string_lossy().into_owned());
        progress.bar.set_message(format!("Splitting {name} "));
        progress.bar.set_style(style);
        progress
    }
}

impl Drop for CliProgress {
    fn drop(&mut self) {
        if !self.finished {
            self.bar.finish_and_clear();
        }
    }
}

impl ProgressReporter for CliProgress {
    fn start(&mut self, total: Option<Duration>) {
        self.total = total.map(duration_to_millis);
        if let Some(total_ms) = self.total {
            if total_ms > 0 {
                self.bar.set_length(total_ms);
            }
        }
    }

    fn update(&mut self, processed: Duration) {
        if let Some(total_ms) = self.total {
            let position = duration_to_millis(processed).min(total_ms);
            self.bar.set_position(position);
        } else {
            self.bar.set_message(format!(
                "Splitting (processed {}) ",
                HumanDuration(processed)
            ));
            self.bar.tick();
        }
    }

    fn finish(&mut self) {
        if let Some(total_ms) = self.total {
            self.bar.set_position(total_ms);
        }
        self.finished = true;
        self.bar.finish_and_clear();
    }
}

fn duration_to_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

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
    let dry_run = matches.get_flag("dry-run");

    let config = Config::builder(input_path, output_dir, segment_length, postfix)
        .overwrite(overwrite)
        .build()
        .with_context(|| {
            format!(
                "failed to create configuration for '{}'",
                input_path.display()
            )
        })?;

    if dry_run {
        let segments = audiosplit_core::dry_run(config)
            .with_context(|| format!("failed to simulate split for '{}'", input_path.display()))?;
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
        let mut progress = CliProgress::new(input_path);
        audiosplit_core::run_with_progress(config, &mut progress)
            .with_context(|| format!("failed to split '{}'", input_path.display()))?;
    }

    Ok(())
}
