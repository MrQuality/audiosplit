//! Core logic for splitting audio files into evenly sized chunks.
//!
//! The crate exposes a [`Config`] type for describing how audio should be
//! segmented and a [`run`] function that performs the actual split. Most
//! applications will construct a configuration by canonicalizing the desired
//! input file and destination directory, then invoke [`run`] to emit numbered
//! audio segments. Errors are reported through [`AudioSplitError`], allowing
//! callers to recover from issues such as unsupported codecs or invalid paths.

use log::error;
use std::convert::TryFrom;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::mem;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal, SignalSpec};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::{conv::FromSample, sample::Sample};
use symphonia::default::{get_codecs, get_probe};
use thiserror::Error;

/// Maximum number of segments produced from a single input file.
const MAX_SEGMENTS: u64 = 50_000;
/// Default number of frames pulled from the decoder in a single chunk.
pub const OPTIMAL_DECODE_BUFFER_FRAMES: usize = 4_096;
/// Default number of interleaved samples written to disk in a single chunk.
pub const OPTIMAL_WRITE_BUFFER_SAMPLES: usize = OPTIMAL_DECODE_BUFFER_FRAMES * 2;
/// Backwards-compatible alias for the previous constant name used by the CLI.
pub const DEFAULT_BUFFER_FRAMES: usize = OPTIMAL_DECODE_BUFFER_FRAMES;
/// Backwards-compatible alias for consumers expecting the write buffer constant.
pub const DEFAULT_WRITE_BUFFER_SAMPLES: usize = OPTIMAL_WRITE_BUFFER_SAMPLES;
/// Maximum number of threads allowed when parallelising encoding work.
pub const MAX_THREADS: usize = 32;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Trait used to relay progress information while splitting audio files.
pub trait ProgressReporter {
    /// Called before processing begins. Provides the total duration when known.
    fn start(&mut self, _total_duration: Option<Duration>) {}

    /// Called after progress is made, providing the processed duration so far.
    fn update(&mut self, _processed: Duration) {}

    /// Called once the splitting completes or the operation terminates.
    fn finish(&mut self) {}
}

/// Trait used during dry-run simulations to capture planned segment paths.
pub trait DryRunRecorder {
    /// Record a prospective output path.
    fn record(&mut self, _path: &Path) {}
}

struct NoProgressReporter;

impl ProgressReporter for NoProgressReporter {}

#[derive(Default)]
struct CollectingDryRunRecorder {
    segments: Vec<PathBuf>,
}

impl DryRunRecorder for CollectingDryRunRecorder {
    fn record(&mut self, path: &Path) {
        self.segments.push(path.to_path_buf());
    }
}

enum Execution<'a> {
    Write(SegmentEncoder),
    DryRun(&'a mut dyn DryRunRecorder),
}

impl<'a> Execution<'a> {
    fn write(threads: NonZeroUsize) -> Result<Self, AudioSplitError> {
        SegmentEncoder::new(threads).map(Execution::Write)
    }

    fn complete(&mut self) -> Result<(), AudioSplitError> {
        match self {
            Execution::Write(encoder) => encoder.finish(),
            Execution::DryRun(_) => Ok(()),
        }
    }
}

/// Errors that can occur while splitting audio files.
///
/// # Examples
///
/// ```
/// use audiosplit_core::{AudioSplitError, Config};
/// use std::fs;
/// use std::time::Duration;
/// use tempfile::tempdir;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let temp = tempdir()?;
/// let missing_input = temp.path().join("missing.wav");
/// let output_dir = temp.path().join("segments");
/// fs::create_dir_all(&output_dir)?;
///
/// match Config::new(&missing_input, &output_dir, Duration::from_secs(1), "part") {
///     Err(AudioSplitError::InvalidPath(path)) => assert_eq!(path, missing_input),
///     other => panic!("unexpected result: {other:?}"),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Error)]
pub enum AudioSplitError {
    /// Wrapper around errors produced by the Symphonia decoding library.
    #[error(transparent)]
    Symphonia(#[from] SymphoniaError),

    /// Wrapper around IO errors encountered while reading or writing files.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Error returned when the provided path cannot be resolved.
    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),

    /// Error returned when a generated segment would overwrite an existing file.
    #[error("output file already exists: {0}")]
    OutputExists(PathBuf),

    /// Error returned when the requested segment duration is invalid.
    #[error("invalid segment length: {reason}")]
    InvalidSegmentLength { reason: SegmentLengthError },

    /// Error returned when the number of generated segments exceeds the supported limit.
    #[error("segment limit of {limit} exceeded")]
    SegmentLimitExceeded { limit: u64 },

    /// Error returned when the decoder track lacks a sample rate.
    #[error("input stream does not advertise a sample rate")]
    MissingSampleRate,

    /// Error returned when the container does not expose any default track.
    #[error("input stream does not provide a default track")]
    MissingDefaultTrack,

    /// Error returned when the container format is not supported.
    #[error("unsupported container format")]
    UnsupportedFormat,

    /// Error returned when the codec of the track cannot be handled.
    #[error("unsupported codec")]
    UnsupportedCodec,

    /// Error returned when the decoded sample format cannot be processed.
    #[error("unsupported sample format")]
    UnsupportedSampleFormat,

    /// Error produced when a file name cannot be derived from the input path.
    #[error("failed to derive a base name for the input file")]
    InvalidInputName,

    /// Error returned when the channel configuration cannot be represented.
    #[error("unsupported channel layout")]
    UnsupportedChannelLayout,

    /// Error returned when the output segment exceeds the WAV size limits.
    #[error("segment is too large to be written as a WAV file")]
    SegmentTooLarge,

    /// Error returned when the configured output directory cannot be found.
    #[error("output directory does not exist: {0}")]
    MissingOutputDirectory(PathBuf),

    /// Error returned when files cannot be written to the output directory.
    #[error("output directory is not writable: {0}")]
    OutputDirectoryNotWritable(PathBuf),

    /// Error returned when the destination lacks sufficient free space.
    #[error(
        "insufficient disk space in output directory {path}: required {required} bytes, only {available} bytes available"
    )]
    InsufficientDiskSpace {
        path: PathBuf,
        required: u64,
        available: u64,
    },

    /// Error returned when the configured write buffer cannot hold even a single frame.
    #[error("write buffer of {requested} samples cannot accommodate {channels} audio channels")]
    WriteBufferTooSmall { requested: usize, channels: usize },
}

/// Reasons why a segment length is considered invalid.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum SegmentLengthError {
    /// The configured segment duration was zero or negative.
    #[error("must be greater than zero")]
    Zero,
    /// The segment duration exceeds representable limits.
    #[error("is too large")]
    TooLarge,
}

/// Configuration for the audio splitting operation.
///
/// The configuration ensures paths are canonicalized prior to processing and
/// stores the segment length, filename postfix, and desired degree of parallelism
/// used during splitting.
///
/// # Examples
///
/// ```
/// use audiosplit_core::Config;
/// use std::fs::{self, File};
/// use std::time::Duration;
/// use tempfile::tempdir;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let temp = tempdir()?;
/// let input = temp.path().join("input.wav");
/// File::create(&input)?;
/// let output_dir = temp.path().join("segments");
/// fs::create_dir_all(&output_dir)?;
///
/// let config = Config::new(&input, &output_dir, Duration::from_secs(1), "part")?;
/// assert_eq!(config.segment_length, Duration::from_secs(1));
/// assert_eq!(config.postfix, "part");
/// assert_eq!(config.threads.get(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Config {
    /// Canonicalized path of the source file to split.
    pub input_path: PathBuf,
    /// Canonicalized directory into which the output files will be written.
    pub output_dir: PathBuf,
    /// Desired length of each segment.
    pub segment_length: Duration,
    /// Postfix inserted into the output file names.
    pub postfix: String,
    /// Whether existing output files may be overwritten.
    pub allow_overwrite: bool,
    /// Maximum number of frames buffered in memory before flushing to disk.
    pub buffer_size_frames: NonZeroUsize,
    /// Maximum number of interleaved samples buffered before writing to disk.
    pub write_buffer_samples: NonZeroUsize,
    /// Number of worker threads available for encoding. A value of one performs serial encoding.
    pub threads: NonZeroUsize,
}

impl Config {
    /// Construct a new [`Config`], canonicalizing the provided paths.
    ///
    /// # Parameters
    /// - `input`: Path to the source media file.
    /// - `output`: Directory where the split segments will be written.
    /// - `segment_length`: Desired length of each segment.
    /// - `postfix`: Postfix inserted into the generated filenames.
    ///
    /// # Returns
    /// A canonicalized [`Config`] ready for use with [`run`].
    ///
    /// # Errors
    /// Returns [`AudioSplitError::InvalidSegmentLength`] when `segment_length` is zero.
    /// Returns [`AudioSplitError::InvalidPath`] when `input` is not a readable
    /// file or `output` does not resolve to a directory.
    /// Returns [`AudioSplitError::MissingOutputDirectory`] when the provided
    /// output path does not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use audiosplit_core::Config;
    /// use std::fs::{self, File};
    /// use std::time::Duration;
    /// use tempfile::tempdir;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let temp = tempdir()?;
    /// let input_path = temp.path().join("input.wav");
    /// File::create(&input_path)?;
    /// let output_dir = temp.path().join("segments");
    /// fs::create_dir_all(&output_dir)?;
    ///
    /// let config = Config::new(&input_path, &output_dir, Duration::from_millis(500), "demo")?;
    /// assert_eq!(config.postfix, "demo");
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, S: Into<String>>(
        input: P,
        output: Q,
        segment_length: Duration,
        postfix: S,
    ) -> Result<Self, AudioSplitError> {
        ConfigBuilder::new(input, output, segment_length, postfix).build()
    }

    /// Create a [`ConfigBuilder`] pre-populated with the provided values.
    pub fn builder<P: AsRef<Path>, Q: AsRef<Path>, S: Into<String>>(
        input: P,
        output: Q,
        segment_length: Duration,
        postfix: S,
    ) -> ConfigBuilder {
        ConfigBuilder::new(input, output, segment_length, postfix)
    }
}

/// Builder for constructing [`Config`] instances with additional options.
pub struct ConfigBuilder {
    input: PathBuf,
    output: PathBuf,
    segment_length: Duration,
    postfix: String,
    overwrite: bool,
    create_output_dir: bool,
    buffer_size_frames: NonZeroUsize,
    write_buffer_samples: NonZeroUsize,
    threads: NonZeroUsize,
}

impl ConfigBuilder {
    /// Create a builder using the supplied configuration inputs.
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, S: Into<String>>(
        input: P,
        output: Q,
        segment_length: Duration,
        postfix: S,
    ) -> Self {
        Self {
            input: input.as_ref().to_path_buf(),
            output: output.as_ref().to_path_buf(),
            segment_length,
            postfix: postfix.into(),
            overwrite: false,
            create_output_dir: false,
            buffer_size_frames: NonZeroUsize::new(DEFAULT_BUFFER_FRAMES)
                .expect("default buffer size must be non-zero"),
            write_buffer_samples: NonZeroUsize::new(DEFAULT_WRITE_BUFFER_SAMPLES)
                .expect("default write buffer size must be non-zero"),
            threads: NonZeroUsize::new(1).expect("default thread count must be non-zero"),
        }
    }

    /// Allow or forbid overwriting existing segment files.
    pub fn overwrite(mut self, allow: bool) -> Self {
        self.overwrite = allow;
        self
    }

    /// Allow or forbid creating the output directory when it does not exist.
    pub fn create_output_dir(mut self, allow: bool) -> Self {
        self.create_output_dir = allow;
        self
    }

    /// Configure the number of frames held in memory before flushing to disk.
    pub fn buffer_size_frames(mut self, frames: NonZeroUsize) -> Self {
        self.buffer_size_frames = frames;
        self
    }

    /// Configure the number of interleaved samples buffered before writing to disk.
    pub fn write_buffer_samples(mut self, samples: NonZeroUsize) -> Self {
        self.write_buffer_samples = samples;
        self
    }

    /// Configure the number of worker threads to use when encoding segments.
    pub fn threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// Finalize the builder, validating paths and constructing the [`Config`].
    pub fn build(self) -> Result<Config, AudioSplitError> {
        validate_segment_length(self.segment_length)?;
        let input_path = canonicalize_existing_file(&self.input)?;
        let output_dir = prepare_output_directory(&self.output, self.create_output_dir)?;
        check_write_permission(&output_dir)?;
        let sanitized_postfix = sanitize_filename(&self.postfix);
        enforce_overwrite_rules(&output_dir, &input_path, &sanitized_postfix, self.overwrite)?;

        Ok(Config {
            input_path,
            output_dir,
            segment_length: self.segment_length,
            postfix: sanitized_postfix,
            allow_overwrite: self.overwrite,
            buffer_size_frames: self.buffer_size_frames,
            write_buffer_samples: self.write_buffer_samples,
            threads: self.threads,
        })
    }
}

fn validate_segment_length(segment_length: Duration) -> Result<(), AudioSplitError> {
    if segment_length <= Duration::ZERO {
        Err(AudioSplitError::InvalidSegmentLength {
            reason: SegmentLengthError::Zero,
        })
    } else {
        Ok(())
    }
}

fn canonicalize_existing_file(path: &Path) -> Result<PathBuf, AudioSplitError> {
    let canonical =
        fs::canonicalize(path).map_err(|_| AudioSplitError::InvalidPath(path.to_path_buf()))?;

    if canonical.is_file() {
        Ok(canonical)
    } else {
        Err(AudioSplitError::InvalidPath(canonical))
    }
}

fn ensure_output_directory(path: &Path) -> Result<PathBuf, AudioSplitError> {
    if !path.exists() {
        return Err(AudioSplitError::MissingOutputDirectory(path.to_path_buf()));
    }

    if !path.is_dir() {
        return Err(AudioSplitError::InvalidPath(path.to_path_buf()));
    }

    let canonical =
        fs::canonicalize(path).map_err(|_| AudioSplitError::InvalidPath(path.to_path_buf()))?;

    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(AudioSplitError::InvalidPath(canonical))
    }
}

fn prepare_output_directory(path: &Path, create: bool) -> Result<PathBuf, AudioSplitError> {
    if !path.exists() {
        if create {
            fs::create_dir_all(path).map_err(AudioSplitError::Io)?;
        } else {
            return Err(AudioSplitError::MissingOutputDirectory(path.to_path_buf()));
        }
    }

    ensure_output_directory(path)
}

fn ensure_can_write_file(path: &Path, allow_overwrite: bool) -> Result<(), AudioSplitError> {
    if !allow_overwrite && path.exists() {
        return Err(AudioSplitError::OutputExists(path.to_path_buf()));
    }

    Ok(())
}

fn check_write_permission(path: &Path) -> Result<(), AudioSplitError> {
    const MAX_ATTEMPTS: u32 = 5;

    for attempt in 0..MAX_ATTEMPTS {
        let candidate = path.join(format!(
            ".audiosplit_write_test_{}_{}_{}",
            process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos(),
            attempt
        ));

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                drop(file);
                match fs::remove_file(&candidate) {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) if err.kind() == io::ErrorKind::PermissionDenied => {
                        return Err(AudioSplitError::OutputDirectoryNotWritable(
                            path.to_path_buf(),
                        ));
                    }
                    Err(err) => return Err(AudioSplitError::Io(err)),
                }
                return Ok(());
            }
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                if attempt + 1 == MAX_ATTEMPTS {
                    return Err(AudioSplitError::Io(err));
                }
            }
            Err(err) => match err.kind() {
                io::ErrorKind::PermissionDenied => {
                    return Err(AudioSplitError::OutputDirectoryNotWritable(
                        path.to_path_buf(),
                    ))
                }
                io::ErrorKind::NotFound => {
                    return Err(AudioSplitError::MissingOutputDirectory(path.to_path_buf()))
                }
                _ => return Err(AudioSplitError::Io(err)),
            },
        }
    }

    Err(AudioSplitError::OutputDirectoryNotWritable(
        path.to_path_buf(),
    ))
}

fn ensure_output_dir_present(path: &Path) -> Result<(), AudioSplitError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(AudioSplitError::MissingOutputDirectory(path.to_path_buf()))
    }
}

fn ensure_available_disk_space(
    output_dir: &Path,
    required_bytes: u64,
) -> Result<(), AudioSplitError> {
    if required_bytes == 0 {
        return Ok(());
    }

    let available = query_available_space(output_dir).map_err(AudioSplitError::Io)?;
    if available < required_bytes {
        Err(AudioSplitError::InsufficientDiskSpace {
            path: output_dir.to_path_buf(),
            required: required_bytes,
            available,
        })
    } else {
        Ok(())
    }
}

fn query_available_space(path: &Path) -> io::Result<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let bytes = path.as_os_str().as_bytes();
        let c_path = CString::new(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains null byte"))?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let result = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if result != 0 {
            Err(io::Error::last_os_error())
        } else {
            let block_size = u128::from(stat.f_frsize);
            let available_blocks = u128::from(stat.f_bavail);
            let bytes = block_size.saturating_mul(available_blocks);
            Ok(bytes.min(u128::from(u64::MAX)) as u64)
        }
    }

    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(u64::MAX)
    }
}

fn enforce_overwrite_rules(
    output_dir: &Path,
    input_path: &Path,
    postfix: &str,
    allow_overwrite: bool,
) -> Result<(), AudioSplitError> {
    if allow_overwrite {
        return Ok(());
    }

    let Some(base_name) = input_path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(());
    };
    let Some(extension) = input_path.extension().and_then(|s| s.to_str()) else {
        return Ok(());
    };

    let base_name = sanitize_filename(base_name);
    let prefix = format!("{base_name}_{postfix}_");
    let suffix = format!(".{extension}");

    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }

        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else {
            continue;
        };

        if name.starts_with(&prefix) && name.ends_with(&suffix) {
            return Err(AudioSplitError::OutputExists(entry.path()));
        }
    }

    Ok(())
}

fn frames_to_duration(frames: u64, sample_rate: u64) -> Duration {
    if sample_rate == 0 {
        return Duration::ZERO;
    }

    let rate = u128::from(sample_rate);
    let frames_ns = u128::from(frames).saturating_mul(NANOS_PER_SECOND);
    let total_ns = frames_ns.div_ceil(rate);

    let secs = total_ns / NANOS_PER_SECOND;
    let nanos = (total_ns % NANOS_PER_SECOND) as u32;

    if secs > u128::from(u64::MAX) {
        Duration::MAX
    } else {
        Duration::new(secs as u64, nanos)
    }
}

fn duration_to_frames(duration: Duration, sample_rate: u64) -> Result<u64, AudioSplitError> {
    if sample_rate == 0 {
        return Ok(0);
    }

    let rate = u128::from(sample_rate);
    let secs_frames = rate.checked_mul(u128::from(duration.as_secs())).ok_or(
        AudioSplitError::InvalidSegmentLength {
            reason: SegmentLengthError::TooLarge,
        },
    )?;

    let nanos = u128::from(duration.subsec_nanos());
    let nanos_frames = if nanos == 0 {
        0
    } else {
        let scaled = rate
            .checked_mul(nanos)
            .ok_or(AudioSplitError::InvalidSegmentLength {
                reason: SegmentLengthError::TooLarge,
            })?;
        scaled
            .checked_add(NANOS_PER_SECOND - 1)
            .ok_or(AudioSplitError::InvalidSegmentLength {
                reason: SegmentLengthError::TooLarge,
            })?
            / NANOS_PER_SECOND
    };

    let total_frames =
        secs_frames
            .checked_add(nanos_frames)
            .ok_or(AudioSplitError::InvalidSegmentLength {
                reason: SegmentLengthError::TooLarge,
            })?;

    u64::try_from(total_frames).map_err(|_| AudioSplitError::InvalidSegmentLength {
        reason: SegmentLengthError::TooLarge,
    })
}

fn ensure_segment_limit(
    duration: Duration,
    segment_length: Duration,
) -> Result<(), AudioSplitError> {
    if segment_length.is_zero() {
        return Err(AudioSplitError::InvalidSegmentLength {
            reason: SegmentLengthError::Zero,
        });
    }

    let segment_ns = segment_length.as_nanos();
    if segment_ns == 0 {
        return Err(AudioSplitError::InvalidSegmentLength {
            reason: SegmentLengthError::Zero,
        });
    }

    let total_ns = duration.as_nanos();
    let adjusted =
        total_ns
            .checked_add(segment_ns - 1)
            .ok_or(AudioSplitError::InvalidSegmentLength {
                reason: SegmentLengthError::TooLarge,
            })?;
    let total_segments = adjusted / segment_ns;

    if total_segments > u128::from(MAX_SEGMENTS) {
        Err(AudioSplitError::SegmentLimitExceeded {
            limit: MAX_SEGMENTS,
        })
    } else {
        Ok(())
    }
}

/// Perform the splitting operation using the supplied [`Config`].
///
/// # Parameters
/// - `config`: The fully constructed [`Config`] controlling the split.
///
/// # Returns
/// Returns `Ok(())` once all segments have been written to disk.
///
/// # Errors
/// Propagates [`AudioSplitError`] variants for issues encountered during
/// decoding, validation, or file I/O.
///
/// # Examples
///
/// ```
/// use audiosplit_core::{run, Config};
/// use std::fs::{self, File};
/// use std::io::{Seek, Write};
/// use std::path::Path;
/// use std::time::Duration;
/// use tempfile::tempdir;
///
/// fn write_sine_wav(path: &Path) -> std::io::Result<()> {
///     let sample_rate = 8_000u32;
///     let channels = 1u16;
///     let bits_per_sample = 16u16;
///     let duration_ms = 1_000u32;
///     let total_samples = (sample_rate as u64 * duration_ms as u64 / 1_000) as usize;
///     let mut samples = Vec::with_capacity(total_samples);
///     for n in 0..total_samples {
///         let value = (n as f32 * 440.0 * std::f32::consts::TAU / sample_rate as f32).sin();
///         samples.push((value * i16::MAX as f32 * 0.1) as i16);
///     }
///
///     let mut file = File::create(path)?;
///     let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
///     let block_align = channels * (bits_per_sample / 8);
///     let data_len = (samples.len() * 2) as u32;
///     let chunk_size = 36 + data_len;
///
///     file.write_all(b"RIFF")?;
///     file.write_all(&chunk_size.to_le_bytes())?;
///     file.write_all(b"WAVE")?;
///     file.write_all(b"fmt ")?;
///     file.write_all(&16u32.to_le_bytes())?;
///     file.write_all(&1u16.to_le_bytes())?;
///     file.write_all(&channels.to_le_bytes())?;
///     file.write_all(&sample_rate.to_le_bytes())?;
///     file.write_all(&byte_rate.to_le_bytes())?;
///     file.write_all(&block_align.to_le_bytes())?;
///     file.write_all(&bits_per_sample.to_le_bytes())?;
///     file.write_all(b"data")?;
///     file.write_all(&data_len.to_le_bytes())?;
///     for sample in samples {
///         file.write_all(&sample.to_le_bytes())?;
///     }
///     file.flush()?;
///     file.rewind()?;
///     Ok(())
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let temp = tempdir()?;
/// let input_path = temp.path().join("tone.wav");
/// write_sine_wav(&input_path)?;
/// let output_dir = temp.path().join("segments");
/// fs::create_dir_all(&output_dir)?;
///
/// let config = Config::new(&input_path, &output_dir, Duration::from_millis(250), "part")?;
/// run(config)?;
///
/// let mut produced: Vec<_> = fs::read_dir(&output_dir)?
///     .map(|entry| entry.unwrap().file_name().into_string().unwrap())
///     .collect();
/// produced.sort();
/// assert!(!produced.is_empty());
/// assert!(produced.iter().all(|name| name.contains("part")));
/// # Ok(())
/// # }
/// ```
pub fn run(config: Config) -> Result<(), AudioSplitError> {
    let mut progress = NoProgressReporter;
    let threads = config.threads;
    let mut execution = Execution::write(threads)?;
    run_internal(config, &mut execution, &mut progress).map(|_| ())
}

/// Execute the split operation while reporting progress to the provided reporter.
pub fn run_with_progress<P: ProgressReporter>(
    config: Config,
    progress: &mut P,
) -> Result<(), AudioSplitError> {
    let mut execution = Execution::write(config.threads)?;
    run_internal(config, &mut execution, progress).map(|_| ())
}

/// Execute the split operation and return metrics describing the run.
pub fn run_with_metrics<P: ProgressReporter>(
    config: Config,
    progress: &mut P,
) -> Result<SplitMetrics, AudioSplitError> {
    let mut execution = Execution::write(config.threads)?;
    run_internal(config, &mut execution, progress)
}

/// Perform a dry-run of the split operation, returning the planned segment paths.
pub fn dry_run(config: Config) -> Result<Vec<PathBuf>, AudioSplitError> {
    let mut recorder = CollectingDryRunRecorder::default();
    {
        let mut execution = Execution::DryRun(&mut recorder);
        let mut progress = NoProgressReporter;
        run_internal(config, &mut execution, &mut progress)?;
    }
    Ok(recorder.segments)
}

/// Metrics captured during a streaming split operation.
#[derive(Clone, Debug, Default)]
pub struct SplitMetrics {
    /// Total number of frames processed across all segments.
    pub total_frames_processed: u64,
    /// Number of segments written (or planned during a dry run).
    pub segments_written: u64,
    /// Maximum number of frames buffered at once before flushing to disk.
    pub peak_frames_per_chunk: usize,
    /// Maximum number of interleaved samples buffered at once.
    pub peak_samples_per_chunk: usize,
}

fn run_internal<P: ProgressReporter>(
    config: Config,
    execution: &mut Execution,
    progress: &mut P,
) -> Result<SplitMetrics, AudioSplitError> {
    let mut hint = Hint::new();
    if let Some(extension) = config.input_path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let file = File::open(&config.input_path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = match get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(probed) => probed,
        Err(SymphoniaError::Unsupported(_)) => {
            return Err(AudioSplitError::UnsupportedFormat);
        }
        Err(err) => return Err(AudioSplitError::from(err)),
    };
    let reader = probed.format;

    let track = reader
        .default_track()
        .ok_or(AudioSplitError::MissingDefaultTrack)?;
    if track.codec_params.codec == CODEC_TYPE_NULL {
        return Err(AudioSplitError::UnsupportedCodec);
    }

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(AudioSplitError::MissingSampleRate)? as u64;
    let segment_length_frames = duration_to_frames(config.segment_length, sample_rate)?;
    if segment_length_frames == 0 {
        return Err(AudioSplitError::InvalidSegmentLength {
            reason: SegmentLengthError::Zero,
        });
    }

    let total_duration = track
        .codec_params
        .n_frames
        .map(|total_frames| frames_to_duration(total_frames, sample_rate));
    if let Some(duration) = total_duration {
        ensure_segment_limit(duration, config.segment_length)?;
    }

    progress.start(total_duration);

    let decoder = match get_codecs().make(&track.codec_params, &DecoderOptions::default()) {
        Ok(decoder) => decoder,
        Err(SymphoniaError::Unsupported(_)) => return Err(AudioSplitError::UnsupportedCodec),
        Err(err) => return Err(AudioSplitError::from(err)),
    };

    let base_name = config
        .input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or(AudioSplitError::InvalidInputName)?;
    let base_name = sanitize_filename(base_name);
    let base_name = base_name.to_owned();
    let extension = config
        .input_path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or(AudioSplitError::InvalidInputName)?;
    let extension = extension.to_owned();

    let pad_width = track
        .codec_params
        .n_frames
        .map(|total| {
            let segments = total.div_ceil(segment_length_frames);
            num_width(segments)
        })
        .unwrap_or(1);
    let splitter = StreamingSplitter::new(
        execution,
        &config,
        base_name,
        extension,
        pad_width,
        segment_length_frames,
        sample_rate,
    );

    let run_result = splitter.run(reader, decoder, progress);
    let metrics = match run_result {
        Ok(metrics) => metrics,
        Err(err) => {
            if let Err(finalize_err) = execution.complete() {
                error!("failed to finalize encoding workers after split error: {finalize_err}");
            }
            return Err(err);
        }
    };

    execution.complete()?;

    let duration = frames_to_duration(metrics.total_frames_processed, sample_rate);
    ensure_segment_limit(duration, config.segment_length)?;

    progress.finish();

    Ok(metrics)
}

struct StreamingSplitter<'exec, 'recorder, 'cfg> {
    execution: &'exec mut Execution<'recorder>,
    config: &'cfg Config,
    base_name: String,
    extension: String,
    pad_width: usize,
    segment_length_frames: u64,
    sample_rate: u64,
    buffer_size_frames: usize,
    write_buffer_samples: usize,
    threads: usize,
    frames_in_segment: u64,
    segment_index: u64,
    total_frames_processed: u64,
    active_segment: Option<ActiveSegment>,
    work_buffer: Vec<i16>,
    peak_frames_per_chunk: usize,
    peak_samples_per_chunk: usize,
    segments_created: u64,
}

impl<'exec, 'recorder, 'cfg> StreamingSplitter<'exec, 'recorder, 'cfg> {
    fn new(
        execution: &'exec mut Execution<'recorder>,
        config: &'cfg Config,
        base_name: String,
        extension: String,
        pad_width: usize,
        segment_length_frames: u64,
        sample_rate: u64,
    ) -> Self {
        Self {
            execution,
            config,
            base_name,
            extension,
            pad_width,
            segment_length_frames,
            sample_rate,
            buffer_size_frames: config.buffer_size_frames.get(),
            write_buffer_samples: config.write_buffer_samples.get(),
            threads: config.threads.get(),
            frames_in_segment: 0,
            segment_index: 0,
            total_frames_processed: 0,
            active_segment: None,
            work_buffer: Vec::with_capacity(config.write_buffer_samples.get()),
            peak_frames_per_chunk: 0,
            peak_samples_per_chunk: 0,
            segments_created: 0,
        }
    }

    fn run<P: ProgressReporter>(
        mut self,
        mut reader: Box<dyn FormatReader>,
        mut decoder: Box<dyn Decoder>,
        progress: &mut P,
    ) -> Result<SplitMetrics, AudioSplitError> {
        while let Ok(packet) = reader.next_packet() {
            match decoder.decode(&packet) {
                Ok(decoded) => self.process_decoded_buffer(decoded, progress)?,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(err) => return Err(AudioSplitError::from(err)),
            }
        }

        self.finalize_active_segment()?;

        Ok(SplitMetrics {
            total_frames_processed: self.total_frames_processed,
            segments_written: self.segments_created,
            peak_frames_per_chunk: self.peak_frames_per_chunk,
            peak_samples_per_chunk: self.peak_samples_per_chunk,
        })
    }

    fn process_decoded_buffer<P: ProgressReporter>(
        &mut self,
        decoded: AudioBufferRef<'_>,
        progress: &mut P,
    ) -> Result<(), AudioSplitError> {
        match decoded {
            AudioBufferRef::U8(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::U16(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::U24(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::U32(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::S8(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::S16(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::S24(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::S32(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::F32(buf) => self.process_typed_buffer(buf.as_ref(), progress),
            AudioBufferRef::F64(buf) => self.process_typed_buffer(buf.as_ref(), progress),
        }
    }

    fn process_typed_buffer<S, P>(
        &mut self,
        buffer: &AudioBuffer<S>,
        progress: &mut P,
    ) -> Result<(), AudioSplitError>
    where
        S: Sample + Copy + Send + Sync,
        i16: FromSample<S>,
        P: ProgressReporter,
    {
        let spec = *buffer.spec();
        let channel_count = spec.channels.count();
        if channel_count == 0 {
            return Ok(());
        }

        if self.write_buffer_samples < channel_count {
            return Err(AudioSplitError::WriteBufferTooSmall {
                requested: self.write_buffer_samples,
                channels: channel_count,
            });
        }

        let total_frames = buffer.frames();
        if total_frames == 0 {
            return Ok(());
        }

        let mut channel_slices = Vec::with_capacity(channel_count);
        for channel in 0..channel_count {
            channel_slices.push(buffer.chan(channel));
        }

        let mut frame_index = 0;
        let frames_per_write_chunk = self.write_buffer_samples / channel_count;
        let frames_per_write_chunk = frames_per_write_chunk.max(1);
        while frame_index < total_frames {
            let frames_to_take = self
                .buffer_size_frames
                .min(total_frames.saturating_sub(frame_index))
                .min(frames_per_write_chunk);
            let frames_to_take = frames_to_take.max(1);
            self.peak_frames_per_chunk = self.peak_frames_per_chunk.max(frames_to_take);

            let required_samples = frames_to_take * channel_count;
            let mut samples = mem::take(&mut self.work_buffer);
            if samples.capacity() < required_samples {
                samples.reserve(required_samples - samples.capacity());
            }

            let start_frame = frame_index;
            samples.resize(required_samples, 0);
            if self.threads > 1 {
                self.fill_samples_parallel(
                    &channel_slices,
                    start_frame,
                    frames_to_take,
                    channel_count,
                    &mut samples,
                );
            } else {
                Self::fill_samples_serial(
                    &channel_slices,
                    start_frame,
                    frames_to_take,
                    channel_count,
                    &mut samples,
                );
            }

            self.peak_samples_per_chunk = self.peak_samples_per_chunk.max(samples.len());

            self.write_samples(spec, channel_count, frames_to_take as u64, &samples)?;
            self.work_buffer = samples;
            self.work_buffer.clear();
            frame_index += frames_to_take;
        }

        self.total_frames_processed = self
            .total_frames_processed
            .saturating_add(total_frames as u64);
        let processed_duration = frames_to_duration(self.total_frames_processed, self.sample_rate);
        progress.update(processed_duration);
        Ok(())
    }

    fn fill_samples_parallel<S>(
        &self,
        channel_slices: &[&[S]],
        start_frame: usize,
        frames_to_take: usize,
        channel_count: usize,
        output: &mut [i16],
    ) where
        S: Sample + Copy + Send + Sync,
        i16: FromSample<S>,
    {
        if frames_to_take == 0 {
            return;
        }

        let threads = self.threads.max(1).min(frames_to_take);
        if threads <= 1 {
            Self::fill_samples_serial(
                channel_slices,
                start_frame,
                frames_to_take,
                channel_count,
                output,
            );
            return;
        }

        let mut remaining = output;
        let mut frame_offset = 0usize;
        let mut frames_remaining = frames_to_take;

        thread::scope(|scope| {
            for i in 0..threads {
                let threads_left = threads - i;
                debug_assert!(
                    threads_left > 0,
                    "chunk division requires non-zero threads_left"
                );
                let chunk_frames = frames_remaining.div_ceil(threads_left);
                let split_index = chunk_frames * channel_count;
                let (chunk_slice, rest) = remaining.split_at_mut(split_index);
                remaining = rest;
                let chunk_start = start_frame + frame_offset;

                scope.spawn(move || {
                    Self::fill_samples_serial(
                        channel_slices,
                        chunk_start,
                        chunk_frames,
                        channel_count,
                        chunk_slice,
                    );
                });

                frame_offset += chunk_frames;
                frames_remaining -= chunk_frames;
            }
        });
    }

    fn fill_samples_serial<S>(
        channel_slices: &[&[S]],
        start_frame: usize,
        frames_to_take: usize,
        channel_count: usize,
        output: &mut [i16],
    ) where
        S: Sample + Copy,
        i16: FromSample<S>,
    {
        for (offset, chunk) in output.chunks_mut(channel_count).enumerate() {
            if offset >= frames_to_take {
                break;
            }
            let frame = start_frame + offset;
            for (channel, sample_slot) in chunk.iter_mut().enumerate() {
                *sample_slot = i16::from_sample(channel_slices[channel][frame]);
            }
        }
    }

    fn write_samples(
        &mut self,
        spec: SignalSpec,
        channel_count: usize,
        frames: u64,
        samples: &[i16],
    ) -> Result<(), AudioSplitError> {
        if frames == 0 || channel_count == 0 {
            return Ok(());
        }

        let mut frames_remaining = frames;
        let mut sample_offset: usize = 0;

        while frames_remaining > 0 {
            if self.frames_in_segment >= self.segment_length_frames {
                self.finalize_active_segment()?;
            }

            if self.active_segment.is_none() {
                self.start_new_segment(spec, channel_count)?;
            }

            let remaining_in_segment = self
                .segment_length_frames
                .saturating_sub(self.frames_in_segment);
            if remaining_in_segment == 0 {
                self.finalize_active_segment()?;
                continue;
            }

            let frames_to_write = frames_remaining.min(remaining_in_segment);
            let samples_to_write = frames_to_write as usize * channel_count;
            let end = sample_offset + samples_to_write;
            if let Some(segment) = self.active_segment.as_mut() {
                segment
                    .samples
                    .extend_from_slice(&samples[sample_offset..end]);
            }

            sample_offset = end;
            frames_remaining -= frames_to_write;
            self.frames_in_segment += frames_to_write;

            if self.frames_in_segment >= self.segment_length_frames {
                self.finalize_active_segment()?;
            }
        }

        Ok(())
    }

    fn start_new_segment(
        &mut self,
        spec: SignalSpec,
        channel_count: usize,
    ) -> Result<(), AudioSplitError> {
        if self.segment_index >= MAX_SEGMENTS {
            return Err(AudioSplitError::SegmentLimitExceeded {
                limit: MAX_SEGMENTS,
            });
        }

        self.segment_index += 1;
        self.segments_created += 1;
        self.pad_width = self.pad_width.max(num_width(self.segment_index));

        let writer_params = WriterParams {
            config: self.config,
            base_name: &self.base_name,
            extension: &self.extension,
            pad_width: self.pad_width,
            segment_index: self.segment_index,
            signal_spec: spec,
        };
        match self.execution {
            Execution::Write(_encoder) => {
                let target = prepare_segment_target(&writer_params, channel_count)?;
                self.active_segment = Some(ActiveSegment::new(target, self.write_buffer_samples));
            }
            Execution::DryRun(recorder) => {
                let planned = plan_segment_path(&writer_params)?;
                ensure_can_write_file(&planned, self.config.allow_overwrite)?;
                recorder.record(&planned);
                let channels = u16::try_from(channel_count)
                    .map_err(|_| AudioSplitError::UnsupportedChannelLayout)?;
                let placeholder = SegmentTarget {
                    path: planned,
                    sample_rate: writer_params.signal_spec.rate,
                    channels,
                };
                self.active_segment = Some(ActiveSegment::new(placeholder, 0));
            }
        }
        Ok(())
    }

    fn finalize_active_segment(&mut self) -> Result<(), AudioSplitError> {
        if let Some(segment) = self.active_segment.take() {
            if let Execution::Write(encoder) = self.execution {
                let samples = segment.samples;
                if samples.is_empty() {
                    self.frames_in_segment = 0;
                    return Ok(());
                }

                let data_bytes = (samples.len() as u64)
                    .checked_mul(2)
                    .ok_or(AudioSplitError::SegmentTooLarge)?;
                let header_bytes = 44u64;
                let required_bytes = data_bytes
                    .checked_add(header_bytes)
                    .ok_or(AudioSplitError::SegmentTooLarge)?;
                ensure_available_disk_space(&self.config.output_dir, required_bytes)?;

                let task = SegmentEncodeTask {
                    path: segment.target.path,
                    sample_rate: segment.target.sample_rate,
                    channels: segment.target.channels,
                    samples,
                };
                encoder.submit(task)?;
            }
        }
        self.frames_in_segment = 0;
        Ok(())
    }
}

fn num_width(mut value: u64) -> usize {
    if value == 0 {
        return 1;
    }

    let mut width = 0;
    while value > 0 {
        value /= 10;
        width += 1;
    }
    width
}

struct WriterParams<'a> {
    config: &'a Config,
    base_name: &'a str,
    extension: &'a str,
    pad_width: usize,
    segment_index: u64,
    signal_spec: SignalSpec,
}

fn plan_segment_path(params: &WriterParams) -> Result<PathBuf, AudioSplitError> {
    let padded_index = format!("{:0width$}", params.segment_index, width = params.pad_width);
    let file_name = format!(
        "{}_{}_{}.{}",
        params.base_name, params.config.postfix, padded_index, params.extension
    );

    debug_assert!(params.config.output_dir.is_absolute());

    let mut output_path = params.config.output_dir.clone();
    output_path.push(file_name);

    debug_assert!(output_path.is_absolute());

    if !output_path.starts_with(&params.config.output_dir) {
        return Err(AudioSplitError::InvalidPath(output_path));
    }

    ensure_output_dir_present(&params.config.output_dir)?;

    Ok(output_path)
}

/// Sanitize a user-controlled filename fragment to avoid directory traversal.
///
/// The sanitizer replaces any platform path separators or repeated dot
/// sequences with underscores, ensuring the generated file names cannot escape
/// the canonical output directory. The function always returns at least a
/// single underscore when the input collapses entirely.
fn sanitize_filename(input: &str) -> String {
    let mut sanitized = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if matches!(ch, '/' | '\\') {
            sanitized.push('_');
            continue;
        }

        if ch == '.' && matches!(chars.peek(), Some('.')) {
            while matches!(chars.peek(), Some('.')) {
                chars.next();
            }
            sanitized.push('_');
            continue;
        }

        sanitized.push(ch);
    }

    if sanitized.is_empty() {
        sanitized.push('_');
    }

    sanitized
}

fn prepare_segment_target(
    params: &WriterParams,
    channel_count: usize,
) -> Result<SegmentTarget, AudioSplitError> {
    ensure_write_buffer_can_hold_frame(params.config.write_buffer_samples.get(), channel_count)?;

    let output_path = plan_segment_path(params)?;
    ensure_can_write_file(&output_path, params.config.allow_overwrite)?;

    let channels =
        u16::try_from(channel_count).map_err(|_| AudioSplitError::UnsupportedChannelLayout)?;

    Ok(SegmentTarget {
        path: output_path,
        sample_rate: params.signal_spec.rate,
        channels,
    })
}

fn ensure_write_buffer_can_hold_frame(
    write_buffer_samples: usize,
    channel_count: usize,
) -> Result<(), AudioSplitError> {
    if channel_count == 0 {
        return Err(AudioSplitError::UnsupportedChannelLayout);
    }

    if write_buffer_samples < channel_count {
        return Err(AudioSplitError::WriteBufferTooSmall {
            requested: write_buffer_samples,
            channels: channel_count,
        });
    }

    Ok(())
}

struct SegmentTarget {
    path: PathBuf,
    sample_rate: u32,
    channels: u16,
}

struct ActiveSegment {
    target: SegmentTarget,
    samples: Vec<i16>,
}

impl ActiveSegment {
    fn new(target: SegmentTarget, capacity: usize) -> Self {
        Self {
            target,
            samples: Vec::with_capacity(capacity),
        }
    }
}

struct SegmentEncodeTask {
    path: PathBuf,
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

enum PendingResult {
    Immediate(Result<(), AudioSplitError>),
    Receiver(mpsc::Receiver<Result<(), AudioSplitError>>),
}

struct SegmentEncoder {
    pool: Option<ThreadPool>,
    pending: Vec<PendingResult>,
}

impl SegmentEncoder {
    fn new(threads: NonZeroUsize) -> Result<Self, AudioSplitError> {
        let pool = if threads.get() > 1 {
            Some(ThreadPool::new(threads.get()))
        } else {
            None
        };

        Ok(Self {
            pool,
            pending: Vec::new(),
        })
    }

    fn submit(&mut self, task: SegmentEncodeTask) -> Result<(), AudioSplitError> {
        if let Some(pool) = &self.pool {
            let receiver = pool.submit(task)?;
            self.pending.push(PendingResult::Receiver(receiver));
        } else {
            let result = encode_segment(task);
            self.pending.push(PendingResult::Immediate(result));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), AudioSplitError> {
        let mut first_error: Option<AudioSplitError> = None;

        for pending in self.pending.drain(..) {
            let result = match pending {
                PendingResult::Immediate(result) => result,
                PendingResult::Receiver(receiver) => receiver.recv().unwrap_or_else(|_| {
                    Err(AudioSplitError::Io(io::Error::other(
                        "encoder worker channel closed unexpectedly",
                    )))
                }),
            };

            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }

        if let Some(mut pool) = self.pool.take() {
            pool.shutdown()?;
        }

        if let Some(err) = first_error {
            return Err(err);
        }

        Ok(())
    }
}

struct ThreadPool {
    sender: mpsc::Sender<PoolMessage>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(workers: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let shared_receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(workers);

        for _ in 0..workers {
            let rx = Arc::clone(&shared_receiver);
            let handle = thread::spawn(move || loop {
                let message = {
                    let guard = rx.lock().expect("encoder receiver poisoned");
                    guard.recv()
                };

                match message {
                    Ok(PoolMessage::Task(task, responder)) => {
                        let result = encode_segment(task);
                        let _ = responder.send(result);
                    }
                    Ok(PoolMessage::Shutdown) | Err(_) => break,
                }
            });
            handles.push(handle);
        }

        Self { sender, handles }
    }

    fn submit(
        &self,
        task: SegmentEncodeTask,
    ) -> Result<mpsc::Receiver<Result<(), AudioSplitError>>, AudioSplitError> {
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(PoolMessage::Task(task, sender))
            .map_err(|err| {
                AudioSplitError::Io(io::Error::other(format!(
                    "failed to schedule encoding task: {err}"
                )))
            })?;
        Ok(receiver)
    }

    fn shutdown(&mut self) -> Result<(), AudioSplitError> {
        for _ in &self.handles {
            self.sender.send(PoolMessage::Shutdown).map_err(|err| {
                AudioSplitError::Io(io::Error::other(format!(
                    "failed to signal encoder shutdown: {err}"
                )))
            })?;
        }

        for handle in self.handles.drain(..) {
            handle
                .join()
                .map_err(|_| AudioSplitError::Io(io::Error::other("encoder worker panicked")))?;
        }

        Ok(())
    }
}

enum PoolMessage {
    Task(SegmentEncodeTask, mpsc::Sender<Result<(), AudioSplitError>>),
    Shutdown,
}

fn encode_segment(task: SegmentEncodeTask) -> Result<(), AudioSplitError> {
    let mut file = File::create(task.path)?;
    write_wav_header(&mut file, task.sample_rate, task.channels)?;

    for sample in &task.samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    let data_bytes = (task.samples.len() as u64)
        .checked_mul(2)
        .ok_or(AudioSplitError::SegmentTooLarge)?;
    if data_bytes > u32::MAX as u64 {
        return Err(AudioSplitError::SegmentTooLarge);
    }

    let data_len = data_bytes as u32;
    let chunk_size = 36u32
        .checked_add(data_len)
        .ok_or(AudioSplitError::SegmentTooLarge)?;

    file.seek(SeekFrom::Start(4))?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.seek(SeekFrom::Start(40))?;
    file.write_all(&data_len.to_le_bytes())?;
    file.flush()?;
    Ok(())
}

fn write_wav_header(
    file: &mut File,
    sample_rate: u32,
    channels: u16,
) -> Result<(), AudioSplitError> {
    if channels == 0 {
        return Err(AudioSplitError::UnsupportedChannelLayout);
    }

    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or(AudioSplitError::UnsupportedChannelLayout)?;
    let byte_rate = sample_rate
        .checked_mul(block_align as u32)
        .ok_or(AudioSplitError::UnsupportedChannelLayout)?;

    file.write_all(b"RIFF")?;
    file.write_all(&0u32.to_le_bytes())?; // Placeholder for chunk size
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // PCM header size
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&0u32.to_le_bytes())?; // Placeholder for data length

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use symphonia::core::audio::Channels;
    use tempfile::tempdir;

    #[test]
    fn frames_to_duration_rounds_up_partial_frame() {
        assert_eq!(
            frames_to_duration(48_001, 48_000),
            Duration::from_nanos(1_000_020_834)
        );
    }

    #[test]
    fn frames_to_duration_returns_zero_for_zero_rate() {
        assert_eq!(frames_to_duration(1_000, 0), Duration::ZERO);
    }

    #[test]
    fn duration_to_frames_rejects_excessive_segment_length() {
        let duration = Duration::from_secs(2);
        let sample_rate = u64::MAX;

        let err = duration_to_frames(duration, sample_rate)
            .expect_err("expected conversion to reject overflow");

        match err {
            AudioSplitError::InvalidSegmentLength { reason } => {
                assert_eq!(reason, SegmentLengthError::TooLarge);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_segment_limit_allows_exact_fit() {
        let segment_length = Duration::from_secs(1);
        let duration = Duration::from_secs(MAX_SEGMENTS);

        ensure_segment_limit(duration, segment_length)
            .expect("duration exactly divisible by segment length should pass");
    }

    #[test]
    fn ensure_segment_limit_counts_partial_segment() {
        let segment_length = Duration::from_millis(1_000);
        let duration = segment_length
            .checked_mul(2)
            .and_then(|d| d.checked_add(Duration::from_millis(1)))
            .expect("duration fits in u128");

        ensure_segment_limit(duration, segment_length)
            .expect("partial segments should still be within the limit");
    }

    #[test]
    fn num_width_increases_with_value() {
        assert_eq!(num_width(0), 1);
        assert_eq!(num_width(7), 1);
        assert_eq!(num_width(10), 2);
        assert_eq!(num_width(1_001), 4);
    }

    #[test]
    fn config_new_rejects_zero_length() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let err = Config::new(&input_path, temp_dir.path(), Duration::ZERO, "part")
            .expect_err("expected zero duration rejection");

        match err {
            AudioSplitError::InvalidSegmentLength { reason } => {
                assert_eq!(reason, SegmentLengthError::Zero);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_new_rejects_missing_input() {
        let temp_dir = tempdir().expect("create temp dir");
        let missing_input = temp_dir.path().join("missing.wav");

        let err = Config::new(
            &missing_input,
            temp_dir.path(),
            Duration::from_secs(1),
            "part",
        )
        .expect_err("expected invalid input path error");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, missing_input),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_new_rejects_non_directory_output() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_file = temp_dir.path().join("output_file");
        File::create(&output_file).expect("create output file");

        let err = Config::new(&input_path, &output_file, Duration::from_secs(1), "part")
            .expect_err("expected invalid output directory error");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, output_file),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_new_rejects_missing_output_directory() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments");
        assert!(
            !output_dir.exists(),
            "output directory should start missing"
        );

        let err = Config::new(&input_path, &output_dir, Duration::from_secs(1), "part")
            .expect_err("expected missing output directory error");

        match err {
            AudioSplitError::MissingOutputDirectory(path) => assert_eq!(path, output_dir),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_new_canonicalizes_paths() {
        let temp_dir = tempdir().expect("create temp dir");
        let nested_dir = temp_dir.path().join("nested");
        fs::create_dir_all(&nested_dir).expect("create nested dir");
        let input_path = nested_dir.join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = nested_dir.join("output");
        fs::create_dir_all(&output_dir).expect("create output directory");

        let config = Config::new(&input_path, &output_dir, Duration::from_millis(500), "demo")
            .expect("config should be constructed");

        assert!(config.input_path.is_absolute());
        assert!(config.output_dir.is_absolute());
        assert_eq!(config.segment_length, Duration::from_millis(500));
        assert_eq!(config.postfix, "demo");
    }

    #[test]
    fn config_builder_defaults_to_no_overwrite() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&output_dir).expect("create output directory");

        let config = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .build()
            .expect("config should build");

        assert!(!config.allow_overwrite);
    }

    #[test]
    fn config_builder_rejects_missing_input_file() {
        let temp_dir = tempdir().expect("create temp dir");
        let missing_input = temp_dir.path().join("missing.wav");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&output_dir).expect("create output directory");

        let err = Config::builder(&missing_input, &output_dir, Duration::from_secs(1), "part")
            .build()
            .expect_err("expected missing input rejection");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, missing_input),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_builder_rejects_missing_output_directory_when_creation_disabled() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");
        let output_dir = temp_dir.path().join("missing/output");
        assert!(!output_dir.exists(), "output directory should be missing");

        let err = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .build()
            .expect_err("expected missing output directory rejection");

        match err {
            AudioSplitError::MissingOutputDirectory(path) => assert_eq!(path, output_dir),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_builder_rejects_zero_length_before_touching_output() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");
        let output_dir = temp_dir.path().join("output");

        let err = Config::builder(&input_path, &output_dir, Duration::ZERO, "part")
            .build()
            .expect_err("expected zero duration rejection");

        match err {
            AudioSplitError::InvalidSegmentLength { reason } => {
                assert_eq!(reason, SegmentLengthError::Zero);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(
            !output_dir.exists(),
            "output directory should not be created when duration is invalid"
        );
    }

    #[test]
    fn config_builder_canonicalizes_output_with_parent_components() {
        let temp_dir = tempdir().expect("create temp dir");
        let nested_dir = temp_dir.path().join("nested");
        fs::create_dir_all(&nested_dir).expect("create nested dir");
        let input_path = nested_dir.join("input.wav");
        File::create(&input_path).expect("create input file");

        let relative_output = nested_dir.join("..").join("segments");
        fs::create_dir_all(&relative_output).expect("create output directory");

        let config = Config::builder(
            &input_path,
            &relative_output,
            Duration::from_secs(1),
            "part",
        )
        .build()
        .expect("config should build");

        let expected = fs::canonicalize(&relative_output).expect("canonicalize output");
        assert_eq!(config.output_dir, expected);
    }

    #[test]
    fn config_builder_rejects_output_paths_pointing_to_files() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_file = temp_dir.path().join("not_a_dir");
        File::create(&output_file).expect("create file for output path");

        let err = Config::builder(&input_path, &output_file, Duration::from_secs(1), "part")
            .build()
            .expect_err("expected invalid output path error");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, output_file),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn sanitize_filename_replaces_traversal_sequences() {
        let sanitized = sanitize_filename("..//evil\\name");
        assert_eq!(sanitized, "___evil_name");
        assert!(!sanitized.contains(".."));
        assert!(!sanitized.contains('/'));
        assert!(!sanitized.contains('\\'));
    }

    #[test]
    fn config_builder_sanitizes_postfix_input() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments");
        fs::create_dir_all(&output_dir).expect("create output directory");

        let config = Config::builder(
            &input_path,
            &output_dir,
            Duration::from_secs(1),
            "..//malicious\\suffix..",
        )
        .build()
        .expect("config should sanitize postfix");

        assert!(!config.postfix.contains(".."));
        assert!(!config.postfix.contains('/'));
        assert!(!config.postfix.contains('\\'));
    }

    #[test]
    fn plan_segment_path_never_escapes_output_directory() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("..evil.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments");
        fs::create_dir_all(&output_dir).expect("create output directory");

        let config = Config::builder(
            &input_path,
            &output_dir,
            Duration::from_secs(1),
            "..//malicious\\suffix",
        )
        .build()
        .expect("config should sanitize paths");

        let base_name = sanitize_filename("..evil");
        let params = WriterParams {
            config: &config,
            base_name: &base_name,
            extension: "wav",
            pad_width: 3,
            segment_index: 1,
            signal_spec: SignalSpec::new(48_000, Channels::FRONT_LEFT),
        };

        let planned = plan_segment_path(&params).expect("plan path");
        assert!(planned.starts_with(&config.output_dir));
        let file_name = planned
            .file_name()
            .and_then(|s| s.to_str())
            .expect("valid utf-8 filename");
        assert!(!file_name.contains(".."));
        assert!(!file_name.contains('/'));
        assert!(!file_name.contains('\\'));
    }

    #[test]
    fn config_builder_rejects_existing_segments_without_overwrite() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments");
        fs::create_dir_all(&output_dir).expect("create output directory");
        let existing_segment = output_dir.join("input_part_1.wav");
        File::create(&existing_segment).expect("create existing segment");
        let existing_segment = fs::canonicalize(existing_segment).expect("canonicalize segment");

        let err = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .build()
            .expect_err("expected overwrite prevention");

        match err {
            AudioSplitError::OutputExists(path) => assert_eq!(path, existing_segment),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_builder_allows_existing_segments_with_overwrite_flag() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments");
        fs::create_dir_all(&output_dir).expect("create output directory");
        let existing_segment = output_dir.join("input_part_1.wav");
        File::create(&existing_segment).expect("create existing segment");

        let config = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .overwrite(true)
            .build()
            .expect("overwrite flag should allow existing files");

        assert!(config.allow_overwrite);
        assert!(config.output_dir.ends_with("segments"));
    }

    #[test]
    fn config_builder_creates_output_directory_when_allowed() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("segments/nested");
        assert!(
            !output_dir.exists(),
            "output directory should start missing"
        );

        let config = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .create_output_dir(true)
            .build()
            .expect("builder should create missing directory");

        assert!(
            output_dir.exists(),
            "builder should create directory when allowed"
        );
        assert!(config.output_dir.is_dir());
    }

    #[test]
    fn config_builder_rejects_zero_length_before_creating_output() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");
        let output_dir = temp_dir.path().join("segments");

        let err = Config::builder(&input_path, &output_dir, Duration::ZERO, "part")
            .create_output_dir(true)
            .build()
            .expect_err("expected zero duration rejection");

        match err {
            AudioSplitError::InvalidSegmentLength { reason } => {
                assert_eq!(reason, SegmentLengthError::Zero);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(
            !output_dir.exists(),
            "output directory should not be created when duration is invalid"
        );
    }

    #[test]
    fn ensure_output_directory_rejects_missing_paths() {
        let temp_dir = tempdir().expect("create temp dir");
        let nested_output = temp_dir.path().join("nested/output");

        let err = ensure_output_directory(&nested_output)
            .expect_err("missing directory should be rejected");

        match err {
            AudioSplitError::MissingOutputDirectory(path) => assert_eq!(path, nested_output),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_output_directory_canonicalizes_existing_directory() {
        let temp_dir = tempdir().expect("create temp dir");
        let nested_output = temp_dir.path().join("nested/output");
        fs::create_dir_all(&nested_output).expect("create nested output dir");

        let canonical =
            ensure_output_directory(&nested_output).expect("directory should be accepted");

        assert!(canonical.is_dir());
        assert!(canonical.is_absolute());
    }

    #[test]
    fn ensure_output_directory_rejects_file_paths() {
        let temp_dir = tempdir().expect("create temp dir");
        let file_path = temp_dir.path().join("not_a_dir");
        File::create(&file_path).expect("create file");

        let err =
            ensure_output_directory(&file_path).expect_err("expected rejection for file path");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, file_path),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn check_write_permission_accepts_writable_directory() {
        let temp_dir = tempdir().expect("create temp dir");
        check_write_permission(temp_dir.path()).expect("temporary directory should be writable");
    }

    #[cfg(unix)]
    #[test]
    fn check_write_permission_rejects_read_only_directory() {
        use std::os::unix::fs::PermissionsExt;

        if unsafe { libc::geteuid() } == 0 {
            eprintln!("skipping read-only directory permission check for root user");
            return;
        }

        let temp_dir = tempdir().expect("create temp dir");
        let output_dir = temp_dir.path().join("readonly");
        fs::create_dir_all(&output_dir).expect("create readonly dir");

        let metadata = fs::metadata(&output_dir).expect("read metadata");
        let mut perms = metadata.permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&output_dir, perms).expect("set read-only permissions");

        let err = check_write_permission(&output_dir)
            .expect_err("read-only directory should be rejected");

        match err {
            AudioSplitError::OutputDirectoryNotWritable(path) => assert_eq!(path, output_dir),
            other => panic!("unexpected error variant: {other:?}"),
        }

        let mut perms = fs::metadata(&output_dir)
            .expect("refresh metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&output_dir, perms).expect("restore permissions");
    }

    #[cfg(unix)]
    #[test]
    fn config_builder_rejects_unwritable_output_directory() {
        use std::os::unix::fs::PermissionsExt;

        if unsafe { libc::geteuid() } == 0 {
            eprintln!("skipping unwritable directory test for root user");
            return;
        }

        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("input.wav");
        File::create(&input_path).expect("create input file");

        let output_dir = temp_dir.path().join("readonly");
        fs::create_dir_all(&output_dir).expect("create readonly dir");

        let metadata = fs::metadata(&output_dir).expect("read metadata");
        let mut perms = metadata.permissions();
        perms.set_mode(0o555);
        fs::set_permissions(&output_dir, perms).expect("set read-only permissions");

        let err = Config::builder(&input_path, &output_dir, Duration::from_secs(1), "part")
            .build()
            .expect_err("expected unwritable directory rejection");

        match err {
            AudioSplitError::OutputDirectoryNotWritable(path) => assert_eq!(path, output_dir),
            other => panic!("unexpected error: {other:?}"),
        }

        let mut perms = fs::metadata(&output_dir)
            .expect("refresh metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&output_dir, perms).expect("restore permissions");
    }

    #[test]
    fn ensure_available_disk_space_detects_insufficient_capacity() {
        let temp_dir = tempdir().expect("create temp dir");

        let err = ensure_available_disk_space(temp_dir.path(), u64::MAX)
            .expect_err("available space check should fail for unrealistic requirement");

        match err {
            AudioSplitError::InsufficientDiskSpace {
                path,
                required,
                available,
            } => {
                assert_eq!(path, temp_dir.path());
                assert_eq!(required, u64::MAX);
                assert!(available < required);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_can_write_file_prevents_overwrite_when_disabled() {
        let temp_dir = tempdir().expect("create temp dir");
        let file_path = temp_dir.path().join("segment.wav");
        File::create(&file_path).expect("create file");

        let err =
            ensure_can_write_file(&file_path, false).expect_err("expected overwrite prevention");

        match err {
            AudioSplitError::OutputExists(path) => assert_eq!(path, file_path),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_can_write_file_allows_overwrite_when_enabled() {
        let temp_dir = tempdir().expect("create temp dir");
        let file_path = temp_dir.path().join("segment.wav");
        File::create(&file_path).expect("create file");

        ensure_can_write_file(&file_path, true)
            .expect("overwrite flag should allow existing files");
    }

    #[test]
    fn segment_limit_rejection_from_duration() {
        let segment_length = Duration::from_secs(1);
        let duration = Duration::from_secs(MAX_SEGMENTS)
            .checked_add(Duration::from_secs(1))
            .expect("duration fits within bounds");

        let err = ensure_segment_limit(duration, segment_length)
            .expect_err("expected segment limit rejection");

        match err {
            AudioSplitError::SegmentLimitExceeded { limit } => assert_eq!(limit, MAX_SEGMENTS),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn parallel_encoding_matches_serial_output() {
        let temp_dir = tempdir().expect("create temp dir");
        let input_path = temp_dir.path().join("tone.wav");
        write_test_tone(&input_path, 8_000, 3).expect("generate input tone");

        let serial_output = tempdir().expect("create serial output");
        let parallel_output = tempdir().expect("create parallel output");

        let base_config = Config::builder(
            &input_path,
            serial_output.path(),
            Duration::from_millis(400),
            "part",
        )
        .overwrite(true)
        .build()
        .expect("serial config");

        run(base_config.clone()).expect("serial split");

        let parallel_config = Config::builder(
            &input_path,
            parallel_output.path(),
            Duration::from_millis(400),
            "part",
        )
        .overwrite(true)
        .threads(NonZeroUsize::new(4).expect("non-zero"))
        .build()
        .expect("parallel config");

        run(parallel_config).expect("parallel split");

        let serial_segments = collect_segments(serial_output.path());
        let parallel_segments = collect_segments(parallel_output.path());

        assert_eq!(serial_segments.len(), parallel_segments.len());
        for (serial, parallel) in serial_segments.iter().zip(parallel_segments.iter()) {
            assert_eq!(serial.0, parallel.0, "segment names should match");
            assert_eq!(serial.1, parallel.1, "segment bytes should match");
        }
    }

    fn write_test_tone(path: &Path, sample_rate: u32, seconds: u32) -> io::Result<()> {
        let total_frames = sample_rate as usize * seconds as usize;
        let mut samples = Vec::with_capacity(total_frames);
        for n in 0..total_frames {
            let amplitude = ((n % sample_rate as usize) as f32 / sample_rate as f32 * 2.0 - 1.0)
                * i16::MAX as f32;
            samples.push(amplitude as i16);
        }

        write_wav(path, sample_rate, &samples)
    }

    fn write_wav(path: &Path, sample_rate: u32, samples: &[i16]) -> io::Result<()> {
        let mut file = File::create(path)?;
        let bits_per_sample = 16u16;
        let channels = 1u16;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate as u32 * block_align as u32;
        let data_bytes = samples.len() as u32 * 2;
        let chunk_size = 36u32 + data_bytes;

        file.write_all(b"RIFF")?;
        file.write_all(&chunk_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&bits_per_sample.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_bytes.to_le_bytes())?;

        for sample in samples {
            file.write_all(&sample.to_le_bytes())?;
        }

        Ok(())
    }

    fn collect_segments(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .expect("read output dir")
            .map(|entry| entry.expect("dir entry"))
            .collect();
        entries.sort_by_key(|entry| entry.file_name());

        entries
            .into_iter()
            .map(|entry| {
                let name = entry.file_name().into_string().expect("utf8 name");
                let bytes = fs::read(entry.path()).expect("read segment");
                (name, bytes)
            })
            .collect()
    }
}
