//! Procedural sound generator for placeholder audio assets
//!
//! Run with: cargo run --bin generate_sounds
//!
//! This generates placeholder WAV files for all sound categories.
//! Replace with actual sound assets when available.

use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// WAV file header structure
struct WavHeader {
    sample_rate: u32,
    num_channels: u16,
    bits_per_sample: u16,
    data_size: u32,
}

impl WavHeader {
    fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let byte_rate = self.sample_rate * self.num_channels as u32 * self.bits_per_sample as u32 / 8;
        let block_align = self.num_channels * self.bits_per_sample / 8;
        let file_size = 36 + self.data_size;

        // RIFF header
        writer.write_all(b"RIFF")?;
        writer.write_all(&file_size.to_le_bytes())?;
        writer.write_all(b"WAVE")?;

        // fmt chunk
        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?; // Subchunk size
        writer.write_all(&1u16.to_le_bytes())?; // Audio format (PCM)
        writer.write_all(&self.num_channels.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&self.bits_per_sample.to_le_bytes())?;

        // data chunk header
        writer.write_all(b"data")?;
        writer.write_all(&self.data_size.to_le_bytes())?;

        Ok(())
    }
}

/// Generate a footstep sound
fn generate_footstep(material: &str, variant: u32) -> Vec<i16> {
    let sample_rate = 44100;
    let duration = 0.15; // 150ms
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = vec![0i16; num_samples];

    // Seed randomness with material and variant
    let seed: u32 = material.bytes().map(|b| b as u32).sum::<u32>() + variant * 1000;
    let mut rng = seed;

    // Simple LCG random function
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng >> 16) as f32 / 32768.0) - 1.0
    };

    match material {
        "grass" => {
            // Soft, muffled thump with rustling
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 25.0).exp();
                let noise = rand() * 0.3;
                let thump = (t * 80.0 * PI * 2.0).sin() * 0.7;
                samples[i] = ((thump + noise) * envelope * 20000.0) as i16;
            }
        }
        "stone" => {
            // Hard, clicking impact
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 40.0).exp();
                let click = (t * 400.0 * PI * 2.0).sin() * 0.8;
                let high = (t * 1200.0 * PI * 2.0).sin() * 0.3;
                samples[i] = ((click + high) * envelope * 22000.0) as i16;
            }
        }
        "sand" => {
            // Soft, granular shuffle
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 20.0).exp();
                let noise = rand() * 0.8;
                let low = (t * 60.0 * PI * 2.0).sin() * 0.2;
                samples[i] = ((noise + low) * envelope * 15000.0) as i16;
            }
        }
        "wood" => {
            // Hollow, resonant thump
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 15.0).exp();
                let resonance = (t * 150.0 * PI * 2.0).sin() * 0.6;
                let body = (t * 300.0 * PI * 2.0).sin() * 0.4;
                samples[i] = ((resonance + body) * envelope * 22000.0) as i16;
            }
        }
        "snow" => {
            // Crunchy, compressed sound
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 30.0).exp();
                let crunch = rand() * 0.6;
                let compress = (t * 100.0 * PI * 2.0).sin() * 0.4 * (1.0 - t * 5.0).max(0.0);
                samples[i] = ((crunch + compress) * envelope * 18000.0) as i16;
            }
        }
        "metal" => {
            // Metallic clink
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 50.0).exp();
                let ring = (t * 800.0 * PI * 2.0).sin() * 0.5;
                let high = (t * 2000.0 * PI * 2.0).sin() * 0.3;
                let impact = (t * 200.0 * PI * 2.0).sin() * 0.2;
                samples[i] = ((ring + high + impact) * envelope * 20000.0) as i16;
            }
        }
        "gravel" => {
            // Crunchy, loose rocks
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 25.0).exp();
                let noise = rand() * 0.7;
                let click = if (t * 5000.0) as i32 % 50 < 5 { rand() * 0.5 } else { 0.0 };
                samples[i] = ((noise + click) * envelope * 18000.0) as i16;
            }
        }
        _ => {
            // Generic footstep
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 20.0).exp();
                let thump = (t * 100.0 * PI * 2.0).sin() * 0.7;
                samples[i] = (thump * envelope * 20000.0) as i16;
            }
        }
    }

    samples
}

/// Generate a block break sound
fn generate_break(material: &str, variant: u32) -> Vec<i16> {
    let sample_rate = 44100;
    let duration = 0.25; // 250ms
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = vec![0i16; num_samples];

    let seed: u32 = material.bytes().map(|b| b as u32).sum::<u32>() + variant * 1000;
    let mut rng = seed;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng >> 16) as f32 / 32768.0) - 1.0
    };

    match material {
        "stone" | "generic" => {
            // Hard cracking sound
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 15.0).exp();
                let crack = rand() * 0.5;
                let impact = (t * 200.0 * PI * 2.0).sin() * 0.5 * (-t * 30.0).exp();
                samples[i] = ((crack + impact) * envelope * 25000.0) as i16;
            }
        }
        "wood" => {
            // Splintering sound
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 12.0).exp();
                let splinter = rand() * 0.6;
                let crack = (t * 150.0 * PI * 2.0).sin() * 0.4;
                samples[i] = ((splinter + crack) * envelope * 22000.0) as i16;
            }
        }
        "dirt" | "grass" => {
            // Soft crumbling
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 10.0).exp();
                let crumble = rand() * 0.7;
                let thud = (t * 80.0 * PI * 2.0).sin() * 0.3;
                samples[i] = ((crumble + thud) * envelope * 18000.0) as i16;
            }
        }
        _ => {
            // Generic break
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 15.0).exp();
                let noise = rand() * 0.5;
                samples[i] = (noise * envelope * 20000.0) as i16;
            }
        }
    }

    samples
}

/// Generate a block place sound
fn generate_place(material: &str, variant: u32) -> Vec<i16> {
    let sample_rate = 44100;
    let duration = 0.2; // 200ms
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = vec![0i16; num_samples];

    let seed: u32 = material.bytes().map(|b| b as u32).sum::<u32>() + variant * 1000 + 500;
    let mut rng = seed;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng >> 16) as f32 / 32768.0) - 1.0
    };

    match material {
        "stone" | "generic" => {
            // Solid thunk
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 20.0).exp();
                let thunk = (t * 180.0 * PI * 2.0).sin() * 0.7;
                let click = (t * 600.0 * PI * 2.0).sin() * 0.3 * (-t * 50.0).exp();
                samples[i] = ((thunk + click) * envelope * 22000.0) as i16;
            }
        }
        "wood" => {
            // Hollow knock
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 18.0).exp();
                let knock = (t * 200.0 * PI * 2.0).sin() * 0.6;
                let resonance = (t * 400.0 * PI * 2.0).sin() * 0.4;
                samples[i] = ((knock + resonance) * envelope * 22000.0) as i16;
            }
        }
        "dirt" | "grass" => {
            // Soft plop
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 15.0).exp();
                let plop = (t * 100.0 * PI * 2.0).sin() * 0.7;
                let noise = rand() * 0.3;
                samples[i] = ((plop + noise) * envelope * 18000.0) as i16;
            }
        }
        _ => {
            // Generic place
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let envelope = (-t * 20.0).exp();
                let thunk = (t * 150.0 * PI * 2.0).sin() * 0.7;
                samples[i] = (thunk * envelope * 20000.0) as i16;
            }
        }
    }

    samples
}

/// Save samples as WAV file
fn save_wav(path: &Path, samples: &[i16]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let header = WavHeader {
        sample_rate: 44100,
        num_channels: 1,
        bits_per_sample: 16,
        data_size: (samples.len() * 2) as u32,
    };

    header.write(&mut writer)?;

    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}

fn main() {
    println!("Generating placeholder sound assets...\n");

    // Ensure directories exist
    fs::create_dir_all("assets/sfx/footstep").ok();
    fs::create_dir_all("assets/sfx/break").ok();
    fs::create_dir_all("assets/sfx/place").ok();
    fs::create_dir_all("assets/sfx/ambient").ok();

    let materials = ["grass", "stone", "sand", "wood", "snow", "metal", "gravel", "generic"];
    let variants = 2;

    // Generate footstep sounds
    println!("Generating footstep sounds...");
    for material in &materials {
        for v in 1..=variants {
            let samples = generate_footstep(material, v);
            let path = format!("assets/sfx/footstep/{}-{}.wav", material, v);
            if let Err(e) = save_wav(Path::new(&path), &samples) {
                eprintln!("  Error creating {}: {}", path, e);
            } else {
                println!("  Created: {}", path);
            }
        }
    }

    // Generate break sounds
    println!("\nGenerating break sounds...");
    let break_materials = ["stone", "wood", "dirt", "generic"];
    for material in &break_materials {
        for v in 1..=variants {
            let samples = generate_break(material, v);
            let path = format!("assets/sfx/break/{}-{}.wav", material, v);
            if let Err(e) = save_wav(Path::new(&path), &samples) {
                eprintln!("  Error creating {}: {}", path, e);
            } else {
                println!("  Created: {}", path);
            }
        }
    }

    // Generate place sounds
    println!("\nGenerating place sounds...");
    let place_materials = ["stone", "wood", "dirt", "generic"];
    for material in &place_materials {
        for v in 1..=variants {
            let samples = generate_place(material, v);
            let path = format!("assets/sfx/place/{}-{}.wav", material, v);
            if let Err(e) = save_wav(Path::new(&path), &samples) {
                eprintln!("  Error creating {}: {}", path, e);
            } else {
                println!("  Created: {}", path);
            }
        }
    }

    println!("\nDone! Generated {} sound files.",
        materials.len() * variants as usize +
        break_materials.len() * variants as usize +
        place_materials.len() * variants as usize
    );
    println!("\nNote: These are procedural placeholders. Replace with real sounds for production.");
}
