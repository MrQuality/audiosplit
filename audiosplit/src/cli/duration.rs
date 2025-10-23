use std::fmt;
use std::time::Duration;

/// Parse human-friendly duration strings composed from signed, whitespace- or
/// underscore-delimited `<number><unit>` pairs into [`Duration`].
///
/// # Grammar
///
/// ```text
/// duration  = [ sign ] separators? component ( separators component )* ;
/// sign      = "+" | "-" ;
/// component = number separators? unit ;
/// number    = digits [ "." digits ] ;
/// digits    = digit , { digit } ;
/// unit      = "ms" | "s" | "m" | "h" | "d" ;
/// separators = { whitespace | "_" } ;
/// ```
///
/// Each unit may appear at most once; specifying the same unit multiple times
/// is rejected to avoid ambiguous precedence rules. Fractional numbers are
/// supported up to the precision implied by the target unit. Negative totals
/// and zero durations are rejected.
pub fn parse_duration(value: &str) -> Result<Duration, DurationParseError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(DurationParseError::Empty);
    }

    if let Some((offset, ch)) = trimmed.char_indices().find(|(_, c)| !c.is_ascii()) {
        return Err(DurationParseError::UnexpectedChar {
            index: offset,
            found: ch,
        });
    }

    let bytes = trimmed.as_bytes();
    let len = bytes.len();
    let mut index = 0usize;

    let mut negative = false;
    if index < len {
        match bytes[index] {
            b'+' => {
                index += 1;
            }
            b'-' => {
                negative = true;
                index += 1;
            }
            _ => {}
        }
    }

    skip_separators(bytes, &mut index);

    if index >= len {
        return Err(DurationParseError::ExpectedNumber { index, found: None });
    }

    let mut seen_units = [false; Unit::COUNT];
    let mut total_nanos: u128 = 0;
    let mut saw_component = false;

    while index < len {
        skip_separators(bytes, &mut index);
        if index >= len {
            break;
        }

        let (mantissa, scale, number_len) = parse_number(bytes, index)?;
        index += number_len;

        skip_separators(bytes, &mut index);
        if index >= len {
            return Err(DurationParseError::ExpectedUnit { index, found: None });
        }

        let (unit, unit_len) = parse_unit(trimmed, bytes, index)?;
        index += unit_len;

        let unit_index = unit.index();
        if std::mem::replace(&mut seen_units[unit_index], true) {
            return Err(DurationParseError::DuplicateUnit { unit });
        }

        let max_scale = unit.max_scale();
        if scale > max_scale {
            return Err(DurationParseError::FractionalTooPrecise {
                unit,
                max_precision: max_scale,
            });
        }

        let pow10 =
            POW10
                .get(scale as usize)
                .copied()
                .ok_or(DurationParseError::FractionalTooPrecise {
                    unit,
                    max_precision: max_scale,
                })?;

        let product = mantissa
            .checked_mul(unit.nanos())
            .ok_or(DurationParseError::Overflow)?;

        if product % pow10 != 0 {
            return Err(DurationParseError::FractionalTooPrecise {
                unit,
                max_precision: max_scale,
            });
        }

        let component = product / pow10;

        total_nanos = total_nanos
            .checked_add(component)
            .ok_or(DurationParseError::Overflow)?;
        saw_component = true;
    }

    if !saw_component {
        return Err(DurationParseError::ExpectedNumber {
            index: 0,
            found: trimmed.bytes().next().map(|b| b as char),
        });
    }

    if negative {
        return Err(DurationParseError::NegativeTotal);
    }

    if total_nanos == 0 {
        return Err(DurationParseError::Zero);
    }

    let max_nanos = u128::from(u64::MAX) * 1_000_000u128;
    if total_nanos > max_nanos {
        return Err(DurationParseError::TooLarge);
    }

    let secs = (total_nanos / 1_000_000_000) as u64;
    let nanos = (total_nanos % 1_000_000_000) as u32;

    Ok(Duration::new(secs, nanos))
}

fn parse_number(bytes: &[u8], mut index: usize) -> Result<(u128, u32, usize), DurationParseError> {
    if index >= bytes.len() {
        return Err(DurationParseError::ExpectedNumber { index, found: None });
    }

    if !bytes[index].is_ascii_digit() {
        return Err(DurationParseError::ExpectedNumber {
            index,
            found: Some(bytes[index] as char),
        });
    }

    let mut mantissa: u128 = 0;
    let mut scale: u32 = 0;
    let mut seen_decimal = false;
    let mut decimal_index = None;
    let mut consumed = 0usize;

    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'0'..=b'9' => {
                mantissa = mantissa
                    .checked_mul(10)
                    .ok_or(DurationParseError::Overflow)?;
                mantissa = mantissa
                    .checked_add((byte - b'0') as u128)
                    .ok_or(DurationParseError::Overflow)?;
                if seen_decimal {
                    scale = scale.checked_add(1).ok_or(DurationParseError::Overflow)?;
                }
                index += 1;
                consumed += 1;
            }
            b'.' if !seen_decimal => {
                seen_decimal = true;
                decimal_index = Some(index);
                index += 1;
                consumed += 1;
            }
            b'.' => {
                return Err(DurationParseError::UnexpectedChar { index, found: '.' });
            }
            _ => break,
        }
    }

    if seen_decimal && scale == 0 {
        let dot_index = decimal_index.unwrap_or(index);
        return Err(DurationParseError::MissingFractionDigits { index: dot_index });
    }

    Ok((mantissa, scale, consumed))
}

fn parse_unit(
    original: &str,
    bytes: &[u8],
    index: usize,
) -> Result<(Unit, usize), DurationParseError> {
    let remaining = &bytes[index..];
    if remaining.starts_with(b"ms") {
        return Ok((Unit::Millisecond, 2));
    }
    if remaining.starts_with(b"s") {
        return Ok((Unit::Second, 1));
    }
    if remaining.starts_with(b"m") {
        return Ok((Unit::Minute, 1));
    }
    if remaining.starts_with(b"h") {
        return Ok((Unit::Hour, 1));
    }
    if remaining.starts_with(b"d") {
        return Ok((Unit::Day, 1));
    }

    if index >= bytes.len() {
        return Err(DurationParseError::ExpectedUnit { index, found: None });
    }

    let mut end = index;
    while end < bytes.len() && bytes[end].is_ascii_alphabetic() {
        end += 1;
    }

    if end > index {
        let invalid = &original[index..end];
        return Err(DurationParseError::UnknownUnit {
            index,
            found: invalid.to_string(),
        });
    }

    Err(DurationParseError::ExpectedUnit {
        index,
        found: Some(bytes[index] as char),
    })
}

fn skip_separators(bytes: &[u8], index: &mut usize) {
    while *index < bytes.len() {
        match bytes[*index] {
            b'_' => *index += 1,
            b if b.is_ascii_whitespace() => *index += 1,
            _ => break,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurationParseError {
    Empty,
    ExpectedNumber { index: usize, found: Option<char> },
    ExpectedUnit { index: usize, found: Option<char> },
    UnknownUnit { index: usize, found: String },
    DuplicateUnit { unit: Unit },
    MissingFractionDigits { index: usize },
    FractionalTooPrecise { unit: Unit, max_precision: u32 },
    UnexpectedChar { index: usize, found: char },
    NegativeTotal,
    Zero,
    Overflow,
    TooLarge,
}

impl std::error::Error for DurationParseError {}

impl fmt::Display for DurationParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DurationParseError::Empty => write!(f, "duration cannot be empty"),
            DurationParseError::ExpectedNumber { index, found } => match found {
                Some(ch) => write!(
                    f,
                    "expected a number at position {} but found '{}'",
                    index + 1,
                    ch
                ),
                None => write!(f, "expected a number at position {}", index + 1),
            },
            DurationParseError::ExpectedUnit { index, found } => match found {
                Some(ch) => write!(
                    f,
                    "expected a unit at position {} but found '{}'",
                    index + 1,
                    ch
                ),
                None => write!(f, "expected a unit at position {}", index + 1),
            },
            DurationParseError::UnknownUnit { index, found } => {
                write!(f, "unknown unit '{}' at position {}", found, index + 1)
            }
            DurationParseError::DuplicateUnit { unit } => write!(
                f,
                "unit '{}' appears multiple times; durations may not repeat units",
                unit.symbol()
            ),
            DurationParseError::MissingFractionDigits { index } => write!(
                f,
                "expected digits after decimal point at position {}",
                index + 1
            ),
            DurationParseError::FractionalTooPrecise {
                unit,
                max_precision,
            } => write!(
                f,
                "fractional precision for '{}' is limited to {} digits",
                unit.symbol(),
                max_precision
            ),
            DurationParseError::UnexpectedChar { index, found } => write!(
                f,
                "unexpected character '{}' at position {}",
                found,
                index + 1
            ),
            DurationParseError::NegativeTotal => write!(f, "duration cannot be negative"),
            DurationParseError::Zero => write!(f, "duration must be greater than zero"),
            DurationParseError::Overflow => write!(f, "duration component is too large"),
            DurationParseError::TooLarge => write!(
                f,
                "duration exceeds the maximum of {} milliseconds",
                u64::MAX
            ),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Unit {
    Millisecond = 0,
    Second = 1,
    Minute = 2,
    Hour = 3,
    Day = 4,
}

impl Unit {
    const COUNT: usize = 5;

    fn index(self) -> usize {
        self as usize
    }

    fn nanos(self) -> u128 {
        match self {
            Unit::Millisecond => 1_000_000,
            Unit::Second => 1_000_000_000,
            Unit::Minute => 60 * 1_000_000_000,
            Unit::Hour => 3_600 * 1_000_000_000,
            Unit::Day => 86_400 * 1_000_000_000,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Unit::Millisecond => "ms",
            Unit::Second => "s",
            Unit::Minute => "m",
            Unit::Hour => "h",
            Unit::Day => "d",
        }
    }

    fn max_scale(self) -> u32 {
        let mut value = self.nanos();
        let mut zeros = 0u32;
        while value.is_multiple_of(10) {
            value /= 10;
            zeros += 1;
        }
        zeros
    }
}

const POW10: [u128; 13] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_duration(input: &str, expected: Duration) {
        let actual = parse_duration(input).unwrap();
        assert_eq!(actual, expected, "input: {input}");
    }

    #[test]
    fn parses_canonical_inputs() {
        assert_duration("500ms", Duration::from_millis(500));
        assert_duration("30s", Duration::from_secs(30));
        assert_duration("1h30m", Duration::from_secs(5_400));
        assert_duration("2d4h", Duration::from_secs(187_200));
    }

    #[test]
    fn accepts_whitespace_and_underscores() {
        assert_duration("  +1h 30m", Duration::from_secs(5_400));
        assert_duration("1h_30m", Duration::from_secs(5_400));
    }

    #[test]
    fn rejects_uppercase_units() {
        let err = parse_duration("10S").unwrap_err();
        assert!(matches!(err, DurationParseError::UnknownUnit { .. }));
    }

    #[test]
    fn rejects_invalid_formats() {
        assert!(matches!(
            parse_duration("1x"),
            Err(DurationParseError::UnknownUnit { .. })
        ));
        assert!(matches!(
            parse_duration("m30"),
            Err(DurationParseError::ExpectedNumber { .. })
        ));
    }

    #[test]
    fn parses_fractional_components() {
        assert_duration("0.5h", Duration::from_secs(1_800));
        assert_duration("1.25s", Duration::from_nanos(1_250_000_000));
        assert_duration("0.0000000001m", Duration::from_nanos(6));
        assert_duration("0.00000000001h", Duration::from_nanos(36));
    }

    #[test]
    fn rejects_repeated_units() {
        assert!(matches!(
            parse_duration("1h1h"),
            Err(DurationParseError::DuplicateUnit { unit: Unit::Hour })
        ));
    }

    #[test]
    fn rejects_negative_and_zero_durations() {
        assert!(matches!(
            parse_duration("-1s"),
            Err(DurationParseError::NegativeTotal)
        ));
        assert!(matches!(
            parse_duration("0ms"),
            Err(DurationParseError::Zero)
        ));
    }

    #[test]
    fn detects_overflow() {
        let max_ms = u128::from(u64::MAX);
        let overflow_ms = format!("{}ms", max_ms + 1);
        assert!(matches!(
            parse_duration(&overflow_ms),
            Err(DurationParseError::TooLarge)
        ));
    }

    #[test]
    fn rejects_excessive_fractional_precision() {
        assert!(matches!(
            parse_duration("0.1234567ms"),
            Err(DurationParseError::FractionalTooPrecise {
                unit: Unit::Millisecond,
                ..
            })
        ));
    }

    #[test]
    fn rejects_trailing_decimal_points() {
        assert!(matches!(
            parse_duration("1."),
            Err(DurationParseError::MissingFractionDigits { .. })
        ));
    }

    #[test]
    fn validates_component_boundaries() {
        assert!(matches!(
            parse_duration("1h 30"),
            Err(DurationParseError::ExpectedUnit { .. })
        ));
    }
}
