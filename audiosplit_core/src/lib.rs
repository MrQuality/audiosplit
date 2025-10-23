//! Core logic for splitting audio files into evenly sized chunks.
//!
//! The crate exposes a [`Config`] type for describing how audio should be
//! segmented and a [`run`] function that performs the actual split. Most
//! applications will construct a configuration by canonicalizing the desired
//! input file and destination directory, then invoke [`run`] to emit numbered
//! audio segments. Errors are reported through [`AudioSplitError`], allowing
//! callers to recover from issues such as unsupported codecs or invalid paths.

use std::convert::TryFrom;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{mpsc, Arc, Mutex};
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
    Write,
    DryRun(&'a mut dyn DryRunRecorder),
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

    /// Error returned when a background worker fails while encoding a segment.
    #[error("segment encoding worker thread panicked")]
    WorkerThreadPanicked,

    /// Error returned when work cannot be scheduled because the worker pool has shut down.
    #[error("segment encoding workers are unavailable")]
    WorkerPoolShutDown,
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
/// stores the segment length and filename postfix used during splitting.
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
    /// Number of worker threads used to encode completed segments.
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
            threads: NonZeroUsize::new(1).expect("thread count default must be non-zero"),
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

    /// Configure the number of worker threads used to encode completed segments.
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
        enforce_overwrite_rules(&output_dir, &input_path, &self.postfix, self.overwrite)?;

        Ok(Config {
            input_path,
            output_dir,
            segment_length: self.segment_length,
            postfix: self.postfix,
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
    let mut execution = Execution::Write;
    run_internal(config, &mut execution, &mut progress).map(|_| ())
}

/// Execute the split operation while reporting progress to the provided reporter.
pub fn run_with_progress<P: ProgressReporter>(
    config: Config,
    progress: &mut P,
) -> Result<(), AudioSplitError> {
    let mut execution = Execution::Write;
    run_internal(config, &mut execution, progress).map(|_| ())
}

/// Execute the split operation and return metrics describing the run.
pub fn run_with_metrics<P: ProgressReporter>(
    config: Config,
    progress: &mut P,
) -> Result<SplitMetrics, AudioSplitError> {
    let mut execution = Execution::Write;
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

struct SegmentEncodeJob {
    writer: SegmentWriter,
    chunks: Vec<Vec<i16>>,
}

impl SegmentEncodeJob {
    fn new(writer: SegmentWriter, chunks: Vec<Vec<i16>>) -> Self {
        Self { writer, chunks }
    }

    fn execute(mut self) -> Result<(), AudioSplitError> {
        for chunk in self.chunks {
            self.writer.write_samples(&chunk)?;
        }
        self.writer.finalize()
    }
}

struct ActiveSegment {
    writer: SegmentWriter,
    chunks: Vec<Vec<i16>>,
}

impl ActiveSegment {
    fn new(writer: SegmentWriter) -> Self {
        Self {
            writer,
            chunks: Vec::new(),
        }
    }

    fn into_job(self) -> SegmentEncodeJob {
        SegmentEncodeJob::new(self.writer, self.chunks)
    }

    fn push_chunk(&mut self, chunk: Vec<i16>) {
        self.chunks.push(chunk);
    }
}

struct SegmentDispatcher {
    strategy: SegmentDispatchStrategy,
}

impl SegmentDispatcher {
    fn new(thread_count: NonZeroUsize) -> Self {
        if thread_count.get() <= 1 {
            Self {
                strategy: SegmentDispatchStrategy::Inline,
            }
        } else {
            Self {
                strategy: SegmentDispatchStrategy::ThreadPool(ThreadPoolDispatcher::new(
                    thread_count.get(),
                )),
            }
        }
    }

    fn submit(&self, job: SegmentEncodeJob) -> Result<(), AudioSplitError> {
        match &self.strategy {
            SegmentDispatchStrategy::Inline => job.execute(),
            SegmentDispatchStrategy::ThreadPool(pool) => pool.submit(job),
        }
    }

    fn finish(&mut self) -> Result<(), AudioSplitError> {
        match &mut self.strategy {
            SegmentDispatchStrategy::Inline => Ok(()),
            SegmentDispatchStrategy::ThreadPool(pool) => pool.finish(),
        }
    }
}

enum SegmentDispatchStrategy {
    Inline,
    ThreadPool(ThreadPoolDispatcher),
}

struct ThreadPoolDispatcher {
    sender: Mutex<Option<mpsc::Sender<SegmentEncodeJob>>>,
    error_receiver: mpsc::Receiver<AudioSplitError>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ThreadPoolDispatcher {
    fn new(workers: usize) -> Self {
        let (job_sender, job_receiver) = mpsc::channel::<SegmentEncodeJob>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (error_sender, error_receiver) = mpsc::channel();

        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let receiver = Arc::clone(&job_receiver);
            let error_tx = error_sender.clone();
            let handle = std::thread::Builder::new()
                .name(format!("audiosplit-segment-{index}"))
                .spawn(move || loop {
                    let job = {
                        let guard = receiver.lock().expect("receiver mutex poisoned");
                        guard.recv()
                    };
                    match job {
                        Ok(job) => {
                            if let Err(err) = job.execute() {
                                let _ = error_tx.send(err);
                            }
                        }
                        Err(_) => break,
                    }
                })
                .expect("failed to spawn segment worker");
            handles.push(handle);
        }

        drop(error_sender);

        Self {
            sender: Mutex::new(Some(job_sender)),
            error_receiver,
            handles,
        }
    }

    fn submit(&self, job: SegmentEncodeJob) -> Result<(), AudioSplitError> {
        let sender = {
            let guard = self.sender.lock().expect("sender mutex poisoned");
            guard.clone()
        };

        if let Some(sender) = sender {
            if sender.send(job).is_err() {
                if let Ok(err) = self.error_receiver.try_recv() {
                    return Err(err);
                }
                return Err(AudioSplitError::WorkerPoolShutDown);
            }

            if let Ok(err) = self.error_receiver.try_recv() {
                return Err(err);
            }

            Ok(())
        } else {
            Err(AudioSplitError::WorkerPoolShutDown)
        }
    }

    fn finish(&mut self) -> Result<(), AudioSplitError> {
        if let Some(sender) = self.sender.lock().expect("sender mutex poisoned").take() {
            drop(sender);
        }

        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                return Err(AudioSplitError::WorkerThreadPanicked);
            }
        }

        if let Ok(err) = self.error_receiver.try_recv() {
            return Err(err);
        }

        Ok(())
    }
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
    let dispatcher = SegmentDispatcher::new(config.threads);
    let naming = SegmentNaming {
        base_name,
        extension,
        pad_width,
    };
    let splitter = StreamingSplitter::new(
        execution,
        &config,
        naming,
        segment_length_frames,
        sample_rate,
        dispatcher,
    );

    let metrics = splitter.run(reader, decoder, progress)?;

    let duration = frames_to_duration(metrics.total_frames_processed, sample_rate);
    ensure_segment_limit(duration, config.segment_length)?;

    progress.finish();

    Ok(metrics)
}

struct SegmentNaming {
    base_name: String,
    extension: String,
    pad_width: usize,
}

struct StreamingSplitter<'exec, 'recorder, 'cfg> {
    execution: &'exec mut Execution<'recorder>,
    config: &'cfg Config,
    naming: SegmentNaming,
    segment_length_frames: u64,
    sample_rate: u64,
    buffer_size_frames: usize,
    write_buffer_samples: usize,
    frames_in_segment: u64,
    segment_index: u64,
    total_frames_processed: u64,
    dispatcher: SegmentDispatcher,
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
        naming: SegmentNaming,
        segment_length_frames: u64,
        sample_rate: u64,
        dispatcher: SegmentDispatcher,
    ) -> Self {
        Self {
            execution,
            config,
            naming,
            segment_length_frames,
            sample_rate,
            buffer_size_frames: config.buffer_size_frames.get(),
            write_buffer_samples: config.write_buffer_samples.get(),
            frames_in_segment: 0,
            segment_index: 0,
            total_frames_processed: 0,
            dispatcher,
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
        let run_result = (|| -> Result<SplitMetrics, AudioSplitError> {
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
        })();

        let finish_result = self.dispatcher.finish();

        match (run_result, finish_result) {
            (Ok(metrics), Ok(())) => Ok(metrics),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
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
        S: Sample + Copy,
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
            if self.frames_in_segment >= self.segment_length_frames {
                self.finalize_active_segment()?;
            }

            let remaining_in_segment = self
                .segment_length_frames
                .saturating_sub(self.frames_in_segment);
            if remaining_in_segment == 0 {
                self.finalize_active_segment()?;
                continue;
            }

            let remaining_in_segment = usize::try_from(remaining_in_segment).unwrap_or(usize::MAX);
            let mut frames_to_take = self
                .buffer_size_frames
                .min(total_frames.saturating_sub(frame_index))
                .min(frames_per_write_chunk)
                .min(remaining_in_segment);

            if frames_to_take == 0 {
                frames_to_take = remaining_in_segment.min(total_frames.saturating_sub(frame_index));
            }

            let frames_to_take = frames_to_take.max(1);
            self.peak_frames_per_chunk = self.peak_frames_per_chunk.max(frames_to_take);

            let required_samples = frames_to_take * channel_count;
            if self.work_buffer.capacity() < required_samples {
                self.work_buffer
                    .reserve(required_samples - self.work_buffer.capacity());
            }
            self.work_buffer.clear();

            for frame in frame_index..frame_index + frames_to_take {
                for channel_slice in &channel_slices {
                    let sample = channel_slice[frame];
                    self.work_buffer.push(i16::from_sample(sample));
                }
            }

            self.peak_samples_per_chunk = self.peak_samples_per_chunk.max(self.work_buffer.len());

            let samples = self.work_buffer.split_off(0);
            self.write_samples(spec, channel_count, frames_to_take as u64, samples)?;
            frame_index += frames_to_take;
        }

        self.total_frames_processed = self
            .total_frames_processed
            .saturating_add(total_frames as u64);
        let processed_duration = frames_to_duration(self.total_frames_processed, self.sample_rate);
        progress.update(processed_duration);
        Ok(())
    }

    fn write_samples(
        &mut self,
        spec: SignalSpec,
        channel_count: usize,
        frames: u64,
        samples: Vec<i16>,
    ) -> Result<(), AudioSplitError> {
        if frames == 0 || channel_count == 0 || samples.is_empty() {
            return Ok(());
        }

        self.ensure_active_segment(spec)?;

        if let Some(active) = self.active_segment.as_mut() {
            active.push_chunk(samples);
        }

        self.frames_in_segment = self.frames_in_segment.saturating_add(frames);

        if self.frames_in_segment >= self.segment_length_frames {
            self.finalize_active_segment()?;
        }

        Ok(())
    }

    fn ensure_active_segment(&mut self, spec: SignalSpec) -> Result<(), AudioSplitError> {
        if self.active_segment.is_none() {
            self.start_new_segment(spec)?;
        }
        Ok(())
    }

    fn start_new_segment(&mut self, spec: SignalSpec) -> Result<(), AudioSplitError> {
        if self.segment_index >= MAX_SEGMENTS {
            return Err(AudioSplitError::SegmentLimitExceeded {
                limit: MAX_SEGMENTS,
            });
        }

        self.segment_index += 1;
        self.segments_created += 1;
        self.naming.pad_width = self.naming.pad_width.max(num_width(self.segment_index));

        let writer_params = WriterParams {
            config: self.config,
            base_name: &self.naming.base_name,
            extension: &self.naming.extension,
            pad_width: self.naming.pad_width,
            segment_index: self.segment_index,
            signal_spec: spec,
            segment_length_frames: self.segment_length_frames,
        };
        self.active_segment =
            create_writer(&writer_params, self.execution)?.map(ActiveSegment::new);
        self.frames_in_segment = 0;
        Ok(())
    }

    fn finalize_active_segment(&mut self) -> Result<(), AudioSplitError> {
        if let Some(active) = self.active_segment.take() {
            let job = active.into_job();
            match self.execution {
                Execution::Write => self.dispatcher.submit(job)?,
                Execution::DryRun(_) => {
                    job.execute()?;
                }
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
    segment_length_frames: u64,
}

fn create_writer(
    params: &WriterParams,
    execution: &mut Execution,
) -> Result<Option<SegmentWriter>, AudioSplitError> {
    let padded_index = format!("{:0width$}", params.segment_index, width = params.pad_width);
    let file_name = format!(
        "{}_{}_{}.{}",
        params.base_name, params.config.postfix, padded_index, params.extension
    );

    let mut output_path = params.config.output_dir.clone();
    output_path.push(file_name);

    if !output_path.starts_with(&params.config.output_dir) {
        return Err(AudioSplitError::InvalidPath(output_path));
    }

    ensure_output_dir_present(&params.config.output_dir)?;
    ensure_can_write_file(&output_path, params.config.allow_overwrite)?;

    match execution {
        Execution::Write => {
            let channel_count = params.signal_spec.channels.count();
            let channels = u16::try_from(channel_count)
                .map_err(|_| AudioSplitError::UnsupportedChannelLayout)?;

            let bytes_per_frame = channel_count
                .checked_mul(2)
                .ok_or(AudioSplitError::UnsupportedChannelLayout)?
                as u64;
            let required_bytes = params
                .segment_length_frames
                .checked_mul(bytes_per_frame)
                .ok_or(AudioSplitError::InvalidSegmentLength {
                    reason: SegmentLengthError::TooLarge,
                })?;

            ensure_available_disk_space(&params.config.output_dir, required_bytes)?;

            SegmentWriter::create(output_path, params.signal_spec.rate, channels).map(Some)
        }
        Execution::DryRun(recorder) => {
            recorder.record(&output_path);
            Ok(Some(SegmentWriter::dry()))
        }
    }
}

enum SegmentWriter {
    Real(RealSegmentWriter),
    Dry,
}

impl SegmentWriter {
    fn create<P: AsRef<Path>>(
        path: P,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, AudioSplitError> {
        RealSegmentWriter::create(path, sample_rate, channels).map(SegmentWriter::Real)
    }

    fn dry() -> Self {
        SegmentWriter::Dry
    }

    fn write_samples(&mut self, samples: &[i16]) -> Result<(), AudioSplitError> {
        match self {
            SegmentWriter::Real(real) => real.write_samples(samples),
            SegmentWriter::Dry => Ok(()),
        }
    }

    fn finalize(self) -> Result<(), AudioSplitError> {
        match self {
            SegmentWriter::Real(real) => real.finalize(),
            SegmentWriter::Dry => Ok(()),
        }
    }
}

struct RealSegmentWriter {
    file: File,
    data_bytes: u64,
}

impl RealSegmentWriter {
    fn create<P: AsRef<Path>>(
        path: P,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, AudioSplitError> {
        let mut file = File::create(path)?;
        write_wav_header(&mut file, sample_rate, channels)?;
        Ok(Self {
            file,
            data_bytes: 0,
        })
    }

    fn write_samples(&mut self, samples: &[i16]) -> Result<(), AudioSplitError> {
        for &sample in samples {
            let bytes = sample.to_le_bytes();
            self.file.write_all(&bytes)?;
        }
        self.data_bytes = self
            .data_bytes
            .checked_add((samples.len() as u64).saturating_mul(2))
            .ok_or(AudioSplitError::SegmentTooLarge)?;
        Ok(())
    }

    fn finalize(mut self) -> Result<(), AudioSplitError> {
        if self.data_bytes > u32::MAX as u64 {
            return Err(AudioSplitError::SegmentTooLarge);
        }

        let data_len = self.data_bytes as u32;
        let chunk_size = 36u32
            .checked_add(data_len)
            .ok_or(AudioSplitError::SegmentTooLarge)?;

        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&chunk_size.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&data_len.to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
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
}
