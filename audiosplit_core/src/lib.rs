use std::path::Path;

use log::info;
use symphonia::core::errors::Error as SymphoniaError;
use thiserror::Error;

/// Errors that can occur while splitting audio files.
#[derive(Debug, Error)]
pub enum AudioSplitError {
    /// Generic error returned when the functionality is not yet implemented.
    #[error("audio splitting is not yet implemented")]
    Unimplemented,

    /// Wrapper around errors produced by the Symphonia decoding library.
    #[error(transparent)]
    Symphonia(#[from] SymphoniaError),
}

/// Split an input audio file into multiple tracks written to the output directory.
///
/// This is currently a placeholder implementation that only logs the request and
/// returns an [`AudioSplitError::Unimplemented`] error.
pub fn split_audio(input: &Path, output_dir: &Path) -> Result<(), AudioSplitError> {
    info!(
        "requested split of '{}' into '{}'",
        input.display(),
        output_dir.display()
    );

    Err(AudioSplitError::Unimplemented)
}
