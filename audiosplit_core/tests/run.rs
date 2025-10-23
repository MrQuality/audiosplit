use audiosplit_core::{run, run_with_metrics, AudioSplitError, Config, ProgressReporter};
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;
use tempfile::tempdir;

/// Generate lightweight audio fixtures for the tests at runtime.
///
/// Similar to the CLI integration tests, the WAV data is synthesised procedurally
/// so that no binary test assets need to be stored in the repository. A simple
/// sine wave is adequate for exercising the decoding and writing paths.
fn write_test_tone<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    duration_ms: u64,
    channels: u16,
) -> Result<(), Box<dyn Error>> {
    assert!(channels > 0, "channels must be at least 1");
    let total_frames = ((sample_rate as u64 * duration_ms).max(1_000) + 999) / 1_000;
    let mut samples = Vec::with_capacity(total_frames as usize * channels as usize * 2);

    for n in 0..total_frames {
        let theta = (n as f32 / sample_rate as f32) * 2.0 * std::f32::consts::PI * 440.0;
        let sample = (theta.sin() * i16::MAX as f32) as i16;
        for _ in 0..channels {
            samples.extend_from_slice(&sample.to_le_bytes());
        }
    }

    let mut file = File::create(path)?;
    let data_len = samples.len() as u32;
    let chunk_size = 36u32 + data_len;
    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    let bytes_per_sample = 2u16;
    let block_align = bytes_per_sample * channels;
    let byte_rate = sample_rate * block_align as u32;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    file.write_all(&samples)?;
    Ok(())
}

#[test]
fn run_splits_audio_and_keeps_remainder_segment() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("input.wav");
    write_test_tone(&input_path, 8_000, 1_100, 1)?;

    let output_dir = tempdir()?;
    let config = Config::new(
        &input_path,
        output_dir.path(),
        Duration::from_millis(400),
        "chunk",
    )?;
    run(config)?;

    let mut outputs: Vec<_> = fs::read_dir(output_dir.path())?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    outputs.sort();
    assert_eq!(outputs.len(), 3);

    for (index, path) in outputs.iter().enumerate() {
        let file_name = path.file_name().unwrap().to_string_lossy();
        assert!(file_name.starts_with("input_chunk_"));
        assert!(file_name.ends_with(".wav"));
        let expected_suffix = format!("_{}", index + 1);
        assert!(file_name.contains(&expected_suffix));
    }

    let mut sizes: Vec<u64> = outputs
        .iter()
        .map(|path| fs::metadata(path).map(|meta| meta.len()))
        .collect::<Result<_, _>>()?;
    sizes.sort_unstable();
    assert!(sizes.first().unwrap() < sizes.last().unwrap());

    output_dir.close()?;
    work_dir.close()?;
    Ok(())
}

#[test]
fn run_reports_unsupported_format_for_unknown_input() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("input.bin");
    File::create(&input_path)?.write_all(b"not an audio file")?;

    let output_dir = tempdir()?;
    let config = Config::new(
        &input_path,
        output_dir.path(),
        Duration::from_secs(1),
        "part",
    )?;

    let err = run(config).expect_err("unsupported input should fail");
    assert!(matches!(err, AudioSplitError::UnsupportedFormat));

    output_dir.close()?;
    work_dir.close()?;
    Ok(())
}

#[test]
fn run_detects_missing_output_directory() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("tone.wav");
    write_test_tone(&input_path, 8_000, 500, 1)?;

    let output_dir = tempdir()?;
    let output_path = output_dir.path().to_path_buf();
    let config = Config::new(
        &input_path,
        &output_path,
        Duration::from_millis(250),
        "part",
    )?;

    // Remove the directory after configuration has been created to simulate external deletion.
    drop(output_dir);
    assert!(!output_path.exists());

    let err = run(config).expect_err("missing output directory should be reported");
    match err {
        AudioSplitError::MissingOutputDirectory(path) => {
            assert_eq!(path, output_path);
        }
        other => panic!("unexpected error: {other:?}"),
    }

    work_dir.close()?;
    Ok(())
}

#[test]
fn run_enforces_segment_limit() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("long.wav");
    write_test_tone(&input_path, 8_000, 50_001, 1)?;

    let output_dir = tempdir()?;
    let config = Config::new(
        &input_path,
        output_dir.path(),
        Duration::from_millis(1),
        "part",
    )?;
    let err = run(config).expect_err("segment limit should be exceeded");

    match err {
        AudioSplitError::SegmentLimitExceeded { limit } => assert_eq!(limit, 50_000),
        other => panic!("unexpected error: {other:?}"),
    }

    output_dir.close()?;
    work_dir.close()?;
    Ok(())
}

struct SilentProgress;

impl ProgressReporter for SilentProgress {}

#[test]
fn run_limits_buffer_size_usage() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("input.wav");
    write_test_tone(&input_path, 8_000, 1_100, 1)?;

    let output_dir = tempdir()?;
    let buffer_frames = NonZeroUsize::new(128).expect("non-zero");
    let write_buffer_samples = NonZeroUsize::new(32).expect("non-zero");
    let config = Config::builder(
        &input_path,
        output_dir.path(),
        Duration::from_millis(200),
        "chunk",
    )
    .buffer_size_frames(buffer_frames)
    .write_buffer_samples(write_buffer_samples)
    .build()?;

    let mut progress = SilentProgress;
    let metrics = run_with_metrics(config, &mut progress)?;

    assert!(
        metrics.segments_written > 0,
        "expected at least one segment"
    );
    assert!(
        metrics.peak_frames_per_chunk <= buffer_frames.get(),
        "peak frames {} exceeded frame buffer {}",
        metrics.peak_frames_per_chunk,
        buffer_frames
    );
    assert!(
        metrics.peak_frames_per_chunk <= write_buffer_samples.get(),
        "peak frames {} exceeded write buffer {}",
        metrics.peak_frames_per_chunk,
        write_buffer_samples
    );
    assert!(
        metrics.peak_samples_per_chunk <= write_buffer_samples.get(),
        "peak samples {} exceeded write buffer {}",
        metrics.peak_samples_per_chunk,
        write_buffer_samples
    );

    output_dir.close()?;
    work_dir.close()?;
    Ok(())
}

#[test]
fn run_rejects_write_buffer_smaller_than_channel_count() -> Result<(), Box<dyn Error>> {
    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("stereo.wav");
    write_test_tone(&input_path, 8_000, 500, 2)?;

    let output_dir = tempdir()?;
    let buffer_frames = NonZeroUsize::new(32).expect("non-zero");
    let write_buffer_samples = NonZeroUsize::new(1).expect("non-zero");
    let config = Config::builder(
        &input_path,
        output_dir.path(),
        Duration::from_millis(250),
        "chunk",
    )
    .buffer_size_frames(buffer_frames)
    .write_buffer_samples(write_buffer_samples)
    .build()?;

    let err = run(config).expect_err("write buffer smaller than channel count should fail");
    match err {
        AudioSplitError::WriteBufferTooSmall {
            requested,
            channels,
        } => {
            assert_eq!(requested, write_buffer_samples.get());
            assert_eq!(channels, 2);
        }
        other => panic!("unexpected error: {other:?}"),
    }

    output_dir.close()?;
    work_dir.close()?;
    Ok(())
}
