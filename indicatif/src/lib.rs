use std::fmt;
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct ProgressStyle {
    _template: String,
}

#[derive(Debug)]
pub struct TemplateError;

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid progress template")
    }
}

impl std::error::Error for TemplateError {}

impl ProgressStyle {
    pub fn with_template(template: &str) -> Result<Self, TemplateError> {
        Ok(Self {
            _template: template.to_string(),
        })
    }

    pub fn default_bar() -> Self {
        Self {
            _template: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProgressState {
    pub message: String,
    pub length: Option<u64>,
    pub position: u64,
    pub finished: bool,
    pub style: ProgressStyle,
}

#[derive(Clone, Debug)]
pub struct ProgressBar {
    state: ProgressState,
}

impl ProgressBar {
    pub fn new_spinner() -> Self {
        Self {
            state: ProgressState {
                style: ProgressStyle::default_bar(),
                ..ProgressState::default()
            },
        }
    }

    pub fn set_style(&mut self, style: ProgressStyle) {
        self.state.style = style;
    }

    pub fn set_message<S: Into<String>>(&mut self, message: S) {
        self.state.message = message.into();
    }

    pub fn set_length(&mut self, length: u64) {
        self.state.length = Some(length);
    }

    pub fn set_position(&mut self, position: u64) {
        self.state.position = position;
    }

    pub fn tick(&mut self) {}

    pub fn finish_and_clear(&mut self) {
        self.state.finished = true;
    }
}

pub struct HumanDuration(pub Duration);

impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs();
        let millis = self.0.subsec_millis();
        write!(f, "{}.{:03}s", secs, millis)
    }
}
