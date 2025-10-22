mod cli;

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context};
use audiosplit_core::{plan_segments, run_with_progress, Config, ProgressEvent};
use indicatif::{HumanDuration, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::cli::{build_cli, DEFAULT_POSTFIX};

struct ProgressState {
    total_millis: Option<u64>,
    total_label: Option<String>,
}

fn duration_to_millis(duration: Duration) -> u64 {
    duration
        .as_millis()
        .min(u128::from(u64::MAX))
        .try_into()
        .unwrap_or(u64::MAX)
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
        let plan = plan_segments(&config)
            .with_context(|| format!("failed to plan segments for '{}'", input_path.display()))?;

        if plan.is_empty() {
            println!("Dry run: no segments would be generated.");
        } else {
            println!("Dry run: would generate {} segment(s):", plan.len());
            for path in plan {
                println!("  {}", path.display());
            }
        }

        return Ok(());
    }

    let progress = ProgressBar::new(0);
    progress.set_draw_target(ProgressDrawTarget::stderr());

    let bar_style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar());
    let spinner_style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner());

    let state = RefCell::new(ProgressState {
        total_millis: None,
        total_label: None,
    });

    let progress_handle = progress.clone();
    let result = run_with_progress(config, move |event| match event {
        ProgressEvent::Start { total_duration } => {
            let mut state = state.borrow_mut();
            if let Some(duration) = total_duration {
                let total_millis = duration_to_millis(duration).max(1);
                state.total_millis = Some(total_millis);
                state.total_label = Some(format!("{}", HumanDuration(duration)));
                progress_handle.set_style(bar_style.clone());
                progress_handle.set_length(total_millis);
                progress_handle.enable_steady_tick(Duration::from_millis(100));
                if let Some(total_label) = state.total_label.as_deref() {
                    progress_handle.set_message(format!("0 / {total_label}"));
                }
            } else {
                state.total_millis = None;
                state.total_label = None;
                progress_handle.set_style(spinner_style.clone());
                progress_handle.enable_steady_tick(Duration::from_millis(100));
                progress_handle.set_message(String::new());
            }
        }
        ProgressEvent::Advance { processed } => {
            let processed_millis = duration_to_millis(processed);
            let human = format!("{}", HumanDuration(processed));
            let state = state.borrow();
            if let Some(total) = state.total_millis {
                let clamped = processed_millis.min(total);
                progress_handle.set_position(clamped);
                if let Some(total_label) = state.total_label.as_deref() {
                    progress_handle.set_message(format!("{human} / {total_label}"));
                } else {
                    progress_handle.set_message(human);
                }
            } else {
                progress_handle.set_message(human);
            }
        }
        ProgressEvent::Finish => {
            progress_handle.set_message(String::from("Completed"));
        }
    })
    .with_context(|| format!("failed to split '{}'", input_path.display()));

    progress.finish_and_clear();

    result?;

    Ok(())
}
