use std::{env, fs, path::PathBuf};

use image::{ImageFormat, imageops::FilterType};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("../../../assets/icons/icon.jpg");
    let destination = root.join("icons/icon.png");

    println!("cargo:rerun-if-changed={}", source.display());

    let image = image::open(&source).unwrap_or_else(|err| {
        panic!("cannot open source icon {}: {err}", source.display())
    });
    // Preserve the intended square icon shape at a fixed, Tauri-compatible size.
    let icon = image.resize_to_fill(512, 512, FilterType::Lanczos3);

    fs::create_dir_all(destination.parent().expect("icon destination has parent"))
        .expect("cannot create the Tauri icon directory");
    icon.save_with_format(&destination, ImageFormat::Png)
        .expect("cannot convert JPEG icon to PNG");

    tauri_build::build();
}
