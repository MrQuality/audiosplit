//! Core logic for splitting audio files into evenly sized chunks.
//!
//! The crate exposes a [`Config`] type for describing how audio should be
//! segmented and a [`run`] function that performs the actual split. Most
//! applications will construct a configuration by canonicalizing the desired
//! input file and destination directory, then invoke [`run`] to emit numbered
//! audio segments. Errors are reported through [`AudioSplitError`], allowing
//! callers to recover from issues such as unsupported codecs or invalid paths.

use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use thiserror::Error;

/// Maximum number of segments produced from a single input file.
const MAX_SEGMENTS: u64 = 50_000;

/// Errors that can occur while splitting audio files.
///
/// # Examples
///
/// ```
/// use audiosplit_core::{AudioSplitError, Config};
/// use tempfile::tempdir;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let temp = tempdir()?;
/// let missing_input = temp.path().join("missing.wav");
/// let output_dir = temp.path().join("segments");
///
/// match Config::new(&missing_input, &output_dir, 1_000, "part") {
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

    /// Error returned when the requested segment duration is zero.
    #[error("segment length must be greater than zero milliseconds")]
    ZeroDuration,

    /// Error returned when the number of generated segments exceeds the supported limit.
    #[error("segment limit of {limit} exceeded")]
    SegmentLimitExceeded { limit: u64 },

    /// Error returned when the decoder track lacks a sample rate.
    #[error("input stream does not advertise a sample rate")]
    MissingSampleRate,

    /// Error returned when the container does not expose any default track.
    #[error("input stream does not provide a default track")]
    MissingDefaultTrack,

    /// Error returned when the codec of the track cannot be handled.
    #[error("unsupported codec")]
    UnsupportedCodec,

    /// Error produced when a file name cannot be derived from the input path.
    #[error("failed to derive a base name for the input file")]
    InvalidInputName,

    /// Error returned when the channel configuration cannot be represented.
    #[error("unsupported channel layout")]
    UnsupportedChannelLayout,

    /// Error returned when the output segment exceeds the WAV size limits.
    #[error("segment is too large to be written as a WAV file")]
    SegmentTooLarge,
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
/// use tempfile::tempdir;
/// use std::fs::File;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let temp = tempdir()?;
/// let input = temp.path().join("input.wav");
/// File::create(&input)?;
/// let output_dir = temp.path().join("segments");
///
/// let config = Config::new(&input, &output_dir, 1_000, "part")?;
/// assert_eq!(config.segment_length_ms, 1_000);
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
    /// Desired length of each segment in milliseconds.
    pub segment_length_ms: u64,
    /// Postfix inserted into the output file names.
    pub postfix: String,
}

impl Config {
    /// Construct a new [`Config`], canonicalizing the provided paths.
    ///
    /// # Parameters
    /// - `input`: Path to the source media file.
    /// - `output`: Directory where the split segments will be written.
    /// - `segment_length_ms`: Desired length of each segment in milliseconds.
    /// - `postfix`: Postfix inserted into the generated filenames.
    ///
    /// # Returns
    /// A canonicalized [`Config`] ready for use with [`run`].
    ///
    /// # Errors
    /// Returns [`AudioSplitError::ZeroDuration`] when `segment_length_ms` is zero.
    /// Returns [`AudioSplitError::InvalidPath`] when `input` is not a readable
    /// file or `output` cannot be created as a directory.
    ///
    /// # Examples
    ///
    /// ```
    /// use audiosplit_core::Config;
    /// use tempfile::tempdir;
    /// use std::fs::File;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let temp = tempdir()?;
    /// let input_path = temp.path().join("input.wav");
    /// File::create(&input_path)?;
    /// let output_dir = temp.path().join("segments");
    ///
    /// let config = Config::new(&input_path, &output_dir, 500, "demo")?;
    /// assert_eq!(config.postfix, "demo");
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, S: Into<String>>(
        input: P,
        output: Q,
        segment_length_ms: u64,
        postfix: S,
    ) -> Result<Self, AudioSplitError> {
        if segment_length_ms == 0 {
            return Err(AudioSplitError::ZeroDuration);
        }

        let input_ref = input.as_ref();
        let output_ref = output.as_ref();

        let input_path = fs::canonicalize(input_ref)
            .map_err(|_| AudioSplitError::InvalidPath(input_ref.to_path_buf()))?;

        fs::create_dir_all(output_ref)
            .map_err(|_| AudioSplitError::InvalidPath(output_ref.to_path_buf()))?;

        let output_dir = fs::canonicalize(output_ref)
            .map_err(|_| AudioSplitError::InvalidPath(output_ref.to_path_buf()))?;

        if !input_path.is_file() {
            return Err(AudioSplitError::InvalidPath(input_path));
        }

        if !output_dir.is_dir() {
            return Err(AudioSplitError::InvalidPath(output_dir));
        }

        Ok(Self {
            input_path,
            output_dir,
            segment_length_ms,
            postfix: postfix.into(),
        })
    }
}

fn frames_to_milliseconds(frames: u64, sample_rate: u64) -> u64 {
    if sample_rate == 0 {
        return 0;
    }

    frames.saturating_mul(1000).div_ceil(sample_rate)
}

fn ensure_segment_limit(duration_ms: u64, segment_length_ms: u64) -> Result<(), AudioSplitError> {
    if segment_length_ms == 0 {
        return Err(AudioSplitError::ZeroDuration);
    }

    let total_segments = duration_ms.div_ceil(segment_length_ms);

    if total_segments > MAX_SEGMENTS {
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
/// use tempfile::tempdir;
/// use std::fs::{self, File};
/// use std::io::{Seek, Write};
/// use std::path::Path;
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
///
/// let config = Config::new(&input_path, &output_dir, 250, "part")?;
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
    let mut hint = Hint::new();
    if let Some(extension) = config.input_path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let file = File::open(&config.input_path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut reader = probed.format;

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
    let segment_length_frames = sample_rate
        .saturating_mul(config.segment_length_ms)
        .div_ceil(1000);
    if segment_length_frames == 0 {
        return Err(AudioSplitError::ZeroDuration);
    }

    if let Some(total_frames) = track.codec_params.n_frames {
        let duration_ms = frames_to_milliseconds(total_frames, sample_rate);
        ensure_segment_limit(duration_ms, config.segment_length_ms)?;
    }

    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut frames_in_segment: u64 = 0;
    let mut segment_index: u64 = 0;
    let mut total_frames_processed: u64 = 0;

    let base_name = config
        .input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or(AudioSplitError::InvalidInputName)?;
    let extension = config
        .input_path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or(AudioSplitError::InvalidInputName)?;

    let mut pad_width = track
        .codec_params
        .n_frames
        .map(|total| {
            let segments = total.div_ceil(segment_length_frames);
            num_width(segments)
        })
        .unwrap_or(1);

    let mut sample_buffer: Option<SampleBuffer<i16>> = None;
    let mut writer: Option<SegmentWriter> = None;

    while let Ok(packet) = reader.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let total_frames = decoded.frames();
                let capacity = decoded.capacity();

                total_frames_processed = total_frames_processed.saturating_add(total_frames as u64);

                let buffer = sample_buffer
                    .get_or_insert_with(|| SampleBuffer::<i16>::new(capacity as u64, spec));
                buffer.copy_interleaved_ref(decoded);

                let channel_count = spec.channels.count();
                if channel_count == 0 {
                    continue;
                }

                let mut frame_offset: usize = 0;
                let samples = buffer.samples();

                while frame_offset < total_frames {
                    if frames_in_segment >= segment_length_frames {
                        finalize_writer(&mut writer)?;
                        frames_in_segment = 0;
                    }

                    if writer.is_none() {
                        if segment_index >= MAX_SEGMENTS {
                            return Err(AudioSplitError::SegmentLimitExceeded {
                                limit: MAX_SEGMENTS,
                            });
                        }
                        segment_index += 1;
                        pad_width = pad_width.max(num_width(segment_index));
                        writer = Some(create_writer(
                            &config,
                            base_name,
                            extension,
                            pad_width,
                            segment_index,
                            &spec,
                        )?);
                    }

                    let remaining_in_segment =
                        segment_length_frames.saturating_sub(frames_in_segment);
                    let frames_available = total_frames as u64 - frame_offset as u64;
                    let frames_to_write = remaining_in_segment.min(frames_available);

                    let start_sample = frame_offset * channel_count;
                    let end_sample = start_sample + frames_to_write as usize * channel_count;
                    if let Some(active) = writer.as_mut() {
                        active.write_samples(&samples[start_sample..end_sample])?;
                    }

                    frame_offset += frames_to_write as usize;
                    frames_in_segment += frames_to_write;

                    if frames_in_segment >= segment_length_frames {
                        finalize_writer(&mut writer)?;
                        frames_in_segment = 0;
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(AudioSplitError::from(err)),
        }
    }

    finalize_writer(&mut writer)?;

    let duration_ms = frames_to_milliseconds(total_frames_processed, sample_rate);
    ensure_segment_limit(duration_ms, config.segment_length_ms)?;

    Ok(())
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

fn create_writer(
    config: &Config,
    base_name: &str,
    extension: &str,
    pad_width: usize,
    segment_index: u64,
    signal_spec: &SignalSpec,
) -> Result<SegmentWriter, AudioSplitError> {
    let file_name = format!(
        "{}_{}_{}.{extension}",
        base_name,
        config.postfix,
        format_args!("{segment_index:0pad_width$}")
    );

    let mut output_path = config.output_dir.clone();
    output_path.push(file_name);

    if !output_path.starts_with(&config.output_dir) {
        return Err(AudioSplitError::InvalidPath(output_path));
    }

    let channels = u16::try_from(signal_spec.channels.count())
        .map_err(|_| AudioSplitError::UnsupportedChannelLayout)?;

    SegmentWriter::create(output_path, signal_spec.rate, channels)
}

fn finalize_writer(writer: &mut Option<SegmentWriter>) -> Result<(), AudioSplitError> {
    if let Some(active) = writer.take() {
        active.finalize()?;
    }
    Ok(())
}

struct SegmentWriter {
    file: File,
    data_bytes: u64,
}

impl SegmentWriter {
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
    fn frames_to_milliseconds_rounds_up_partial_frame() {
        assert_eq!(frames_to_milliseconds(48_001, 48_000), 1_001);
    }

    #[test]
    fn frames_to_milliseconds_returns_zero_for_zero_rate() {
        assert_eq!(frames_to_milliseconds(1_000, 0), 0);
    }

    #[test]
    fn ensure_segment_limit_allows_exact_fit() {
        let segment_length_ms = 1_000;
        let duration_ms = MAX_SEGMENTS.saturating_mul(segment_length_ms);

        ensure_segment_limit(duration_ms, segment_length_ms)
            .expect("duration exactly divisible by segment length should pass");
    }

    #[test]
    fn ensure_segment_limit_counts_partial_segment() {
        let segment_length_ms = 1_000;
        let duration_ms = segment_length_ms * 2 + 1;

        ensure_segment_limit(duration_ms, segment_length_ms)
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

        let err = Config::new(&input_path, temp_dir.path(), 0, "part")
            .expect_err("expected zero duration rejection");

        assert!(matches!(err, AudioSplitError::ZeroDuration));
    }

    #[test]
    fn config_new_rejects_missing_input() {
        let temp_dir = tempdir().expect("create temp dir");
        let missing_input = temp_dir.path().join("missing.wav");

        let err = Config::new(&missing_input, temp_dir.path(), 1_000, "part")
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

        let err = Config::new(&input_path, &output_file, 1_000, "part")
            .expect_err("expected invalid output directory error");

        match err {
            AudioSplitError::InvalidPath(path) => assert_eq!(path, output_file),
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

        let config = Config::new(&input_path, &output_dir, 500, "demo")
            .expect("config should be constructed");

        assert!(config.input_path.is_absolute());
        assert!(config.output_dir.is_absolute());
        assert_eq!(config.segment_length_ms, 500);
        assert_eq!(config.postfix, "demo");
    }

    #[test]
    fn segment_limit_rejection_from_duration() {
        let segment_length_ms = 1_000;
        let duration_ms = MAX_SEGMENTS
            .saturating_mul(segment_length_ms)
            .saturating_add(1);

        let err = ensure_segment_limit(duration_ms, segment_length_ms)
            .expect_err("expected segment limit rejection");

        match err {
            AudioSplitError::SegmentLimitExceeded { limit } => assert_eq!(limit, MAX_SEGMENTS),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
