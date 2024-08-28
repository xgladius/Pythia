use disasm::collect_functions;
use loader::get_elf_code_section_offset;

use model::predictor::Predictor;
use onnxruntime::environment::Environment;

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::{env, process};

mod disasm;
mod loader;
mod model;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file>", args[0]);
        process::exit(1);
    }

    let file_path = Path::new(&args[1]);

    let mut file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file {}: {}", file_path.display(), e);
            process::exit(1);
        }
    };

    let (offset, size) = get_elf_code_section_offset(&mut file);
    let mut code = vec![0u8; size];
    file.seek(std::io::SeekFrom::Start(offset))
        .expect("Failed to seek file");
    file.read_exact(&mut code).expect("Failed to read file");

    let environment = Environment::builder().build()?;
    let mut predictor = Predictor::new(
        &environment,
        "models/model_start.onnx".into(),
        "models/model_end.onnx".into(),
        "models/tokenizer.json".into(),
    )?;

    let functions = collect_functions(&mut predictor, &code, offset)?;
    println!("{:X?}", functions);
    println!("collected {} functions", functions.len());

    Ok(())
}
