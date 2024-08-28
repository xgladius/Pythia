use std::collections::HashMap;

use iced_x86::{Decoder, DecoderOptions, Formatter, FormatterOutput, IntelFormatter, Mnemonic};
use onnxruntime::session::Session;

use crate::model::predictor::{self, Predictor};

pub struct MyFormatterOutput<'a> {
    buffer: &'a mut String,
}

impl<'a> MyFormatterOutput<'a> {
    pub fn new(buffer: &'a mut String) -> Self {
        Self { buffer }
    }
}

impl<'a> FormatterOutput for MyFormatterOutput<'a> {
    fn write(&mut self, text: &str, _operand_options: iced_x86::FormatterTextKind) {
        self.buffer.push_str(text);
    }
}

struct Function {
    start: u64,
}

impl Function {
    pub fn new() -> Function {
        Function { start: 0 }
    }
}

pub fn decode(predictor: &mut Predictor, code: &[u8], ip: u64) -> anyhow::Result<()> {
    let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
    decoder.set_ip(ip);

    let mut formatter = IntelFormatter::new();
    let mut instructions = Vec::new();
    let mut current_function = Function::new();

    while decoder.can_decode() {
        let instruction = decoder.decode();
        instructions.push(instruction);
    }

    let mut i = 0;
    while i < instructions.len() {
        if instructions[i].mnemonic() == Mnemonic::Nop {
            i += 1;
            continue;
        }

        if i + 4 >= instructions.len() {
            break;
        }

        let cur_candidates = &instructions[i..i + 4];
        let (is_start, start_accuracy) =
            predictor.is_function_start(cur_candidates, &mut formatter);

        if is_start {
            current_function.start = cur_candidates[0].ip();
            println!(
                "found function start ({:.2} confidence) at {:X} for {}",
                start_accuracy,
                cur_candidates[0].ip(),
                cur_candidates
                    .iter()
                    .map(|instr| {
                        let mut output = String::new();
                        formatter.format(instr, &mut MyFormatterOutput::new(&mut output));
                        output
                    })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
            i += cur_candidates.len() + 1;
        } else {
            i += 1;
        }
    }

    Ok(())
}
