use std::path::PathBuf;
use std::time::Duration;

use clap::{builder::ValueParser, value_parser, Arg, ArgAction, Command};

pub const DEFAULT_POSTFIX: &str = "part";

/// Parse a human-friendly duration string into a [`Duration`].
///
/// Supported suffixes are `ms` (milliseconds), `s` (seconds), `m` (minutes),
/// and `h` (hours). Multiple components may be chained together, such as
/// `"1m30s"` or `"2h15m"`. The parser requires the total duration to be
/// greater than zero and representable in whole milliseconds.
pub fn parse_duration(value: &str) -> Result<Duration, String> {
    let input = value.trim();
    if input.is_empty() {
        return Err("duration cannot be empty".into());
    }

    let mut total_ms: u128 = 0;
    let mut index = 0;
    let bytes = input.as_bytes();
    let len = bytes.len();
    let invalid = || format!("invalid duration '{value}'");
    let mut saw_component = false;

    while index < len {
        if bytes[index].is_ascii_whitespace() {
            return Err(invalid());
        }

        let start = index;
        while index < len && bytes[index].is_ascii_digit() {
            index += 1;
        }

        if start == index {
            return Err(invalid());
        }

        let number = input[start..index].parse::<u128>().map_err(|_| invalid())?;

        if index >= len {
            return Err(invalid());
        }

        let remainder = &input[index..];
        let (unit_len, factor) = if remainder.starts_with("ms") {
            (2, 1u128)
        } else if remainder.starts_with('s') {
            (1, 1_000u128)
        } else if remainder.starts_with('m') {
            (1, 60_000u128)
        } else if remainder.starts_with('h') {
            (1, 3_600_000u128)
        } else {
            return Err(invalid());
        };

        index += unit_len;

        let component_ms = number
            .checked_mul(factor)
            .ok_or_else(|| "duration is too large".to_owned())?;
        total_ms = total_ms
            .checked_add(component_ms)
            .ok_or_else(|| "duration is too large".to_owned())?;
        saw_component = true;
    }

    if !saw_component || total_ms == 0 {
        return Err("duration must be greater than zero".into());
    }

    if total_ms > u128::from(u64::MAX) {
        return Err("duration is too large".into());
    }

    Ok(Duration::from_millis(total_ms as u64))
}

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
                .help("Length of each segment (e.g. 500ms, 2m30s)")
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
            Arg::new("dry-run")
                .long("dry-run")
                .help("Preview the generated segments without writing files")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_supports_individual_units() {
        assert_eq!(parse_duration("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_duration("3s").unwrap(), Duration::from_secs(3));
        assert_eq!(parse_duration("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration("1h").unwrap(), Duration::from_secs(3_600));
    }

    #[test]
    fn parse_duration_supports_chained_units() {
        let expected = Duration::from_millis(3_600_000 + 120_000 + 3_000 + 45);
        assert_eq!(parse_duration("1h2m3s45ms").unwrap(), expected);
    }

    #[test]
    fn parse_duration_rejects_missing_units() {
        assert!(parse_duration("100").is_err());
    }

    #[test]
    fn parse_duration_rejects_unknown_units() {
        assert!(parse_duration("5x").is_err());
    }

    #[test]
    fn parse_duration_rejects_zero() {
        assert!(parse_duration("0ms").is_err());
    }
}
