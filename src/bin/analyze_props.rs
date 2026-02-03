//! Analyze prop .vox files for issues

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn main() {
    let props = [
        ("Grass01", "assets/props/nature/grass/grass-01.vox"),
        ("Grass02", "assets/props/nature/grass/grass-02.vox"),
        ("ForestGrass01", "assets/props/nature/grass/forest_grass-01.vox"),
        ("ForestGrass02", "assets/props/nature/grass/forest_grass-02.vox"),
        ("DeadGrass", "assets/props/nature/grass/dead-grass.vox"),
        ("Flower01", "assets/props/nature/flowers/flower-01.vox"),
        ("Flower02", "assets/props/nature/flowers/flower-02.vox"),
        ("Flower03", "assets/props/nature/flowers/flower-03.vox"),
        ("DeadFlower", "assets/props/nature/flowers/dead-flower.vox"),
        ("WaterFlower01", "assets/props/nature/flowers/water-flower-01.vox"),
        ("WaterFlower02", "assets/props/nature/flowers/water-flower-02.vox"),
        ("FlowerSmallPink", "assets/props/nature/flowers/Small/Flower_Small_Pink.vox"),
        ("FlowerSmallRed", "assets/props/nature/flowers/Small/Flower_Small_Red.vox"),
        ("FlowerSmallYellow", "assets/props/nature/flowers/Small/Flower_Small_Yellow.vox"),
        ("FlowerSmallSkyBlue", "assets/props/nature/flowers/Small/Flower_Small_Sky_Blue.vox"),
        ("Stone01", "assets/props/nature/rocks/stone-01.vox"),
        ("Stone02", "assets/props/nature/rocks/stone-02.vox"),
        ("Stone03", "assets/props/nature/rocks/stone-03.vox"),
        ("ForestStone01", "assets/props/nature/rocks/forest_stone-01.vox"),
        ("ForestStone02", "assets/props/nature/rocks/forest_stone-02.vox"),
        ("ForestStone03", "assets/props/nature/rocks/forest_stone-03.vox"),
        ("SandStone", "assets/props/nature/rocks/sand-stone.vox"),
        ("RockFlat", "assets/props/nature/rocks/Rock_Flat.vox"),
        ("Rock", "assets/props/nature/rocks/Rock.vox"),
    ];

    println!("\n=== PROP ANALYSIS ===\n");
    println!("{:<20} {:>8} {:>8} {:>8} {:>10}  {}", "Name", "MinZ", "MaxZ", "Voxels", "Float?", "Top Colors");
    println!("{}", "-".repeat(100));

    for (name, path) in props {
        match analyze_vox(Path::new(path)) {
            Ok((min_z, max_z, voxel_count, colors)) => {
                let floats = if min_z > 0 { format!("YES (+{})", min_z) } else { "no".to_string() };
                let color_str: String = colors.iter()
                    .take(3)
                    .map(|(r,g,b,count)| format!("#{:02X}{:02X}{:02X}({})", r, g, b, count))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("{:<20} {:>8} {:>8} {:>8} {:>10}  {}", name, min_z, max_z, voxel_count, floats, color_str);
            }
            Err(e) => {
                println!("{:<20} ERROR: {}", name, e);
            }
        }
    }
    println!();
}

fn analyze_vox(path: &Path) -> Result<(u8, u8, usize, Vec<(u8,u8,u8,usize)>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"VOX " {
        return Err("Invalid magic".to_string());
    }

    // Skip version
    let mut version = [0u8; 4];
    reader.read_exact(&mut version).ok();

    // Read MAIN
    let mut chunk_id = [0u8; 4];
    reader.read_exact(&mut chunk_id).ok();
    let mut sizes = [0u8; 8];
    reader.read_exact(&mut sizes).ok();

    let mut voxels: Vec<(u8, u8, u8, u8)> = Vec::new();
    let mut palette = [[255u8, 255, 255, 255]; 256];

    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            break;
        }

        let mut content_size = [0u8; 4];
        reader.read_exact(&mut content_size).ok();
        let content_size = u32::from_le_bytes(content_size);

        let mut children_size = [0u8; 4];
        reader.read_exact(&mut children_size).ok();

        match &chunk_id {
            b"XYZI" => {
                let mut num_voxels = [0u8; 4];
                reader.read_exact(&mut num_voxels).ok();
                let num_voxels = u32::from_le_bytes(num_voxels);

                for _ in 0..num_voxels {
                    let mut v = [0u8; 4];
                    reader.read_exact(&mut v).ok();
                    voxels.push((v[0], v[1], v[2], v[3]));
                }
            }
            b"RGBA" => {
                for i in 0..256 {
                    let mut color = [0u8; 4];
                    reader.read_exact(&mut color).ok();
                    let idx = if i == 255 { 0 } else { i + 1 };
                    palette[idx] = color;
                }
            }
            _ => {
                let mut skip = vec![0u8; content_size as usize];
                reader.read_exact(&mut skip).ok();
            }
        }
    }

    if voxels.is_empty() {
        return Err("No voxels".to_string());
    }

    let min_z = voxels.iter().map(|v| v.2).min().unwrap_or(0);
    let max_z = voxels.iter().map(|v| v.2).max().unwrap_or(0);

    // Count colors
    let mut color_counts: HashMap<(u8,u8,u8), usize> = HashMap::new();
    for (_, _, _, color_idx) in &voxels {
        let c = palette[*color_idx as usize];
        *color_counts.entry((c[0], c[1], c[2])).or_insert(0) += 1;
    }

    let mut colors: Vec<_> = color_counts.into_iter()
        .map(|((r,g,b), count)| (r, g, b, count))
        .collect();
    colors.sort_by(|a, b| b.3.cmp(&a.3));

    Ok((min_z, max_z, voxels.len(), colors))
}
