use assert_cmd::Command;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

/// Generate a small single-channel WAV file for testing.
///
/// The fixtures are produced on the fly by emitting a PCM RIFF header followed by
/// procedurally generated sine-wave samples. This keeps the repository free from
/// committed binary assets while still exercising the audio pipeline end-to-end.
fn write_test_tone<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    duration_ms: u64,
) -> Result<(), Box<dyn Error>> {
    let total_samples = ((sample_rate as u64 * duration_ms).max(1_000) + 999) / 1_000;
    let mut samples = Vec::with_capacity(total_samples as usize * 2);

    for n in 0..total_samples {
        let theta = (n as f32 / sample_rate as f32) * 2.0 * std::f32::consts::PI * 440.0;
        let sample = (theta.sin() * i16::MAX as f32) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    let mut file = File::create(path)?;
    let data_len = samples.len() as u32;
    let chunk_size = 36u32 + data_len;
    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // PCM header size
    file.write_all(&1u16.to_le_bytes())?; // audio format = PCM
    file.write_all(&1u16.to_le_bytes())?; // channels
    file.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * 2;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    file.write_all(&samples)?;
    Ok(())
}

#[test]
fn cli_splits_audio_into_segments_with_remainder() -> Result<(), Box<dyn Error>> {
    let input_dir = tempdir()?;
    let input_path = input_dir.path().join("input.wav");
    write_test_tone(&input_path, 8_000, 1_100)?;

    let output_dir = tempdir()?;
    let output_arg = output_dir.path().to_string_lossy().to_string();
    let input_arg = input_path.to_string_lossy().to_string();

    let mut cmd = Command::cargo_bin("audiosplit")?;
    cmd.args(["--length", "400ms", "--output"])
        .arg(&output_arg)
        .arg(&input_arg);
    cmd.assert().success();

    let mut segments: Vec<_> = fs::read_dir(output_dir.path())?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    segments.sort();
    assert_eq!(
        segments.len(),
        3,
        "expected three segments including a remainder"
    );

    let mut sizes: Vec<u64> = segments
        .iter()
        .map(|path| fs::metadata(path).map(|meta| meta.len()))
        .collect::<Result<_, _>>()?;
    sizes.sort_unstable();
    assert!(sizes.first().unwrap() < sizes.last().unwrap());

    output_dir.close()?;
    input_dir.close()?;
    Ok(())
}

#[test]
fn cli_reports_missing_input_file() -> Result<(), Box<dyn Error>> {
    let output_dir = tempdir()?;
    let output_arg = output_dir.path().to_string_lossy().to_string();

    let mut cmd = Command::cargo_bin("audiosplit")?;
    cmd.args(["--length", "400ms", "--output"])
        .arg(&output_arg)
        .arg("missing.wav");
    cmd.assert()
        .failure()
        .stderr_contains("input file does not exist");

    output_dir.close()?;
    Ok(())
}

#[test]
fn cli_dry_run_prints_plan_without_creating_files() -> Result<(), Box<dyn Error>> {
    let input_dir = tempdir()?;
    let input_path = input_dir.path().join("input.wav");
    write_test_tone(&input_path, 8_000, 1_100)?;

    let output_dir = tempdir()?;
    let output_arg = output_dir.path().to_string_lossy().to_string();
    let input_arg = input_path.to_string_lossy().to_string();

    let mut cmd = Command::cargo_bin("audiosplit")?;
    let assert = cmd
        .args(["--length", "400ms", "--output"])
        .arg(&output_arg)
        .arg("--dry-run")
        .arg(&input_arg)
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone())?;
    let expected_paths = [
        output_dir.path().join("input_part_1.wav"),
        output_dir.path().join("input_part_2.wav"),
        output_dir.path().join("input_part_3.wav"),
    ];

    assert!(stdout.contains("Dry run: would generate 3 segment(s):"));
    for path in expected_paths {
        let needle = format!("  {}", path.display());
        assert!(
            stdout.contains(&needle),
            "missing dry-run entry for {needle}"
        );
    }

    let mut produced = std::fs::read_dir(output_dir.path())?;
    assert!(produced.next().is_none(), "dry run should not create files");

    output_dir.close()?;
    input_dir.close()?;
    Ok(())
}
