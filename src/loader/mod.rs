use std::{fs::File, io::Read};

use goblin::elf::Elf;

pub fn get_elf_code_section_offset(file: &mut File) -> (u64, usize) {
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
