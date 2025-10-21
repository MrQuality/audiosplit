use std::collections::VecDeque;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use log::info;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{
    FormatOptions, FormatReader, FormatWriter, FormatWriterOptions, Packet,
};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_formats, get_probe};
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

    /// Wrapper around IO errors encountered while reading or writing files.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Error returned when the decoder track lacks a sample rate.
    #[error("input stream does not advertise a sample rate")]
    MissingSampleRate,

    /// Error returned when the container does not expose any default track.
    #[error("input stream does not provide a default track")]
    MissingDefaultTrack,

    /// Error returned when the codec of the track cannot be handled.
    #[error("unsupported codec")]
    UnsupportedCodec,

    /// Error returned when the segment length is invalid.
    #[error("segment length must be greater than zero milliseconds")]
    InvalidSegmentLength,

    /// Error produced when a file name cannot be derived from the input path.
    #[error("failed to derive a base name for the input file")]
    InvalidInputName,
}

/// Configuration for the audio splitting operation.
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
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>, S: Into<String>>(
        input: P,
        output: Q,
        segment_length_ms: u64,
        postfix: S,
    ) -> Result<Self, AudioSplitError> {
        if segment_length_ms == 0 {
            return Err(AudioSplitError::InvalidSegmentLength);
        }

        let input_path = fs::canonicalize(input)?;
        let output_dir = fs::canonicalize(output)?;

        Ok(Self {
            input_path,
            output_dir,
            segment_length_ms,
            postfix: postfix.into(),
        })
    }
}

/// Perform the splitting operation using the supplied [`Config`].
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
    let segment_length_frames = (sample_rate * config.segment_length_ms + 999) / 1000;
    if segment_length_frames == 0 {
        return Err(AudioSplitError::InvalidSegmentLength);
    }

    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut packets = VecDeque::new();
    let mut frames_in_segment: u64 = 0;
    let mut segment_index: u64 = 0;

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
            let segments = (total + segment_length_frames - 1) / segment_length_frames;
            num_width(segments)
        })
        .unwrap_or(1);

    while let Ok(packet) = reader.next_packet() {
        let packet_ref = packet.clone();
        match decoder.decode(&packet) {
            Ok(decoded) => {
                frames_in_segment += decoded.frames() as u64;
                packets.push_back(packet_ref);
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(AudioSplitError::from(err)),
        }

        if frames_in_segment >= segment_length_frames {
            segment_index += 1;
            pad_width = pad_width.max(num_width(segment_index));
            write_segment(
                &config,
                base_name,
                extension,
                pad_width,
                segment_index,
                &track.codec_params,
                &packets,
            )?;
            frames_in_segment = 0;
            packets.clear();
        }
    }

    if !packets.is_empty() {
        segment_index += 1;
        pad_width = pad_width.max(num_width(segment_index));
        write_segment(
            &config,
            base_name,
            extension,
            pad_width,
            segment_index,
            &track.codec_params,
            &packets,
        )?;
    }

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

fn write_segment(
    config: &Config,
    base_name: &str,
    extension: &str,
    pad_width: usize,
    segment_index: u64,
    params: &CodecParameters,
    packets: &VecDeque<Packet>,
) -> Result<(), AudioSplitError> {
    let file_name = format!(
        "{}_{}_{}.{extension}",
        base_name,
        config.postfix,
        format_args!("{segment_index:0pad_width$}")
    );
    let mut output_path = config.output_dir.clone();
    output_path.push(file_name);

    let mut output = File::create(&output_path)?;
    let mut writer = get_formats().get(probed_format_type(params))?.write(
        &mut output,
        params,
        &FormatWriterOptions::default(),
    )?;

    for packet in packets {
        writer.write(packet)?;
    }
    writer.flush()?;

    Ok(())
}

fn probed_format_type(params: &CodecParameters) -> symphonia::core::formats::FormatId {
    params.codec
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
