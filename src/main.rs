use disasm::decode;
use goblin::elf::Elf;
use model::get_word_index;
use onnxruntime::environment::Environment;
use onnxruntime::GraphOptimizationLevel;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::{env, process};

mod disasm;
mod model;

fn get_code_section_offset(file: &mut File) -> (u64, usize) {
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let elf = match Elf::parse(&buffer) {
        Ok(elf) => elf,
        Err(err) => {
            eprintln!("Failed to parse ELF file: {}", err);
            std::process::exit(1);
        }
    };

    for section in elf.section_headers.iter() {
        if let Some(section_name) = elf.shdr_strtab.get_at(section.sh_name) {
            if section_name == ".text" {
                return (section.sh_offset, section.sh_size as usize);
            }
        }
    }

    eprintln!("No .text section found in the ELF file");
    std::process::exit(1);
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file>", args[0]);
        process::exit(1);
    }

    let file_path = Path::new(&args[1]);

    let environment = Environment::builder().build()?;
    let mut start_session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("models/model_start.onnx")?;

    let word_index = get_word_index("models/tokenizer.json")?;

    let mut file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file {}: {}", file_path.display(), e);
            process::exit(1);
        }
    };

    let (offset, size) = get_code_section_offset(&mut file);
    let mut code = vec![0u8; size];
    file.seek(std::io::SeekFrom::Start(offset))
        .expect("Failed to seek file");
    file.read_exact(&mut code).expect("Failed to read file");

    decode(&mut start_session, &word_index, &code, offset)?;

    Ok(())
}
