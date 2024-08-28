use std::collections::HashMap;

use iced_x86::{
    Decoder, DecoderOptions, Formatter, FormatterOutput, Instruction, IntelFormatter, Mnemonic,
};
use onnxruntime::session::Session;

use crate::model::predict;

struct MyFormatterOutput<'a> {
    buffer: &'a mut String,
}

impl<'a> MyFormatterOutput<'a> {
    fn new(buffer: &'a mut String) -> Self {
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

pub fn decode(
    session: &mut Session<'_>,
    word_index: &HashMap<String, f32>,
    code: &[u8],
    ip: u64,
) -> anyhow::Result<()> {
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
            is_function_start(session, word_index, cur_candidates, &mut formatter);

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

fn is_function_start(
    session: &mut Session<'_>,
    word_index: &HashMap<String, f32>,
    instructions: &[Instruction],
    formatter: &mut IntelFormatter,
) -> (bool, f64) {
    let mut example_input = String::new();
    for instr in instructions.iter() {
        let mut output = String::new();
        formatter.format(instr, &mut MyFormatterOutput::new(&mut output));
        example_input.push_str(&output);
        example_input.push(' '); // Separate instructions with a space
    }

    let start_prediction = predict(session, word_index, example_input, 0.99);

    // Return whether it's a start and a confidence level (for example purposes, 0.99 is used)
    (start_prediction, 0.99)
}
