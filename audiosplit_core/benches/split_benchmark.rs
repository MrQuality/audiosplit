use std::f32::consts::TAU;
use std::fs::File;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use audiosplit_core::{run, Config};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tempfile::TempDir;

struct SyntheticAudio {
    _dir: TempDir,
    path: PathBuf,
}

impl SyntheticAudio {
    fn new(
        file_name: &str,
        sample_rate: u32,
        seconds: u32,
        channels: u16,
        frequency: f32,
    ) -> io::Result<Self> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(file_name);
        write_sine_wave(&path, sample_rate, seconds, channels, frequency)?;
        Ok(Self { _dir: dir, path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

fn write_sine_wave(
    path: &Path,
    sample_rate: u32,
    seconds: u32,
    channels: u16,
    frequency: f32,
) -> io::Result<()> {
    let total_frames = seconds as usize * sample_rate as usize;
    let amplitude = i16::MAX as f32 * 0.6;
    let mut samples = Vec::with_capacity(total_frames * channels as usize);

    for frame in 0..total_frames {
        let t = frame as f32 / sample_rate as f32;
        let sample = (amplitude * (frequency * TAU * t).sin()) as i16;
        for _ in 0..channels {
            samples.push(sample);
        }
    }

    write_wav_pcm_i16(path, sample_rate, channels, &samples)
}

fn write_wav_pcm_i16(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    samples: &[i16],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    let bits_per_sample = 16u16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate as u32 * block_align as u32;
    let data_bytes = (samples.len() * 2) as u32;
    let chunk_size = 36u32 + data_bytes;

    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // PCM header length
    file.write_all(&1u16.to_le_bytes())?; // PCM format
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

struct Scenario {
    name: &'static str,
    segment_length: Duration,
    threads: NonZeroUsize,
}

fn split_benchmarks(c: &mut Criterion) {
    let fixture = SyntheticAudio::new("synthetic.wav", 44_100, 30, 2, 440.0)
        .expect("failed to synthesize audio fixture");

    let scenarios = [
        Scenario {
            name: "segments_1s_serial",
            segment_length: Duration::from_secs(1),
            threads: NonZeroUsize::new(1).expect("thread count must be non-zero"),
        },
        Scenario {
            name: "segments_5s_parallel",
            segment_length: Duration::from_secs(5),
            threads: NonZeroUsize::new(4).expect("thread count must be non-zero"),
        },
        Scenario {
            name: "segments_10s_parallel",
            segment_length: Duration::from_secs(10),
            threads: NonZeroUsize::new(4).expect("thread count must be non-zero"),
        },
    ];

    let mut group = c.benchmark_group("audio_split");

    for scenario in scenarios {
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            &scenario,
            |b, scenario| {
                b.iter_batched(
                    || {
                        let output = tempfile::tempdir().expect("failed to create output dir");
                        let config = Config::builder(
                            fixture.path(),
                            output.path(),
                            scenario.segment_length,
                            "bench",
                        )
                        .overwrite(true)
                        .threads(scenario.threads)
                        .build()
                        .expect("failed to build config");
                        (config, output)
                    },
                    |(config, _output)| {
                        run(config).expect("split run failed");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, split_benchmarks);
criterion_main!(benches);
