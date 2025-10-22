use std::fmt;
use std::time::Duration;

#[derive(Clone, Default)]
pub struct ProgressBar;

impl ProgressBar {
    pub fn new(_length: u64) -> Self {
        Self
    }

    pub fn set_draw_target(&self, _target: ProgressDrawTarget) {}

    pub fn set_style(&self, _style: ProgressStyle) {}

    pub fn set_length(&self, _length: u64) {}

    pub fn enable_steady_tick(&self, _interval: Duration) {}

    pub fn set_message<S: Into<String>>(&self, _msg: S) {}

    pub fn set_position(&self, _pos: u64) {}

    pub fn finish_and_clear(&self) {}
}

#[derive(Clone, Copy, Default)]
pub struct ProgressDrawTarget;

impl ProgressDrawTarget {
    pub fn stderr() -> Self {
        Self
    }
}

#[derive(Clone, Default)]
pub struct ProgressStyle;

impl ProgressStyle {
    pub fn with_template(_template: &str) -> Result<Self, TemplateError> {
        Ok(Self::default())
    }

    pub fn default_bar() -> Self {
        Self::default()
    }

    pub fn default_spinner() -> Self {
        Self::default()
    }
}

#[derive(Debug, Default)]
pub struct TemplateError;

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid template")
    }
}

impl std::error::Error for TemplateError {}

pub struct HumanDuration(pub Duration);

impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs();
        let millis = self.0.subsec_millis();
        if secs == 0 {
            write!(f, "{}ms", millis)
        } else {
            write!(f, "{}.{:03}s", secs, millis)
        }
    }
}
