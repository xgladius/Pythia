use iced_x86::{Decoder, DecoderOptions, Formatter, FormatterOutput, IntelFormatter, Mnemonic};

use crate::model::predictor::Predictor;

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

pub struct Function {
    start: u64,
    end: u64,
}

impl Function {
    pub fn new() -> Function {
        Function { start: 0, end: 0 }
    }
}

pub fn collect_functions(
    predictor: &mut Predictor,
    code: &[u8],
    ip: u64,
) -> anyhow::Result<Vec<Function>> {
    let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
    decoder.set_ip(ip);

    let mut formatter = IntelFormatter::new();
    let mut instructions = Vec::new();
    let mut current_function = Function::new();

    let mut functions = vec![];

    while decoder.can_decode() {
        let instruction = decoder.decode();
        instructions.push(instruction);
    }

    // Collect initial list of functions via RNN inference
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
                "found function start ({:.5} confidence) at {:X} for {}",
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
            functions.push(current_function);
            current_function = Function::new();
            i += cur_candidates.len() + 1;
        } else {
            i += 1;
        }
    }

    // Go over instructions again to collect calls to functions, insert if not already present
    // Naively assumes that the call is to the start of a functions prologue, which is not necessarily true
    let mut i = 0;
    while i < instructions.len() {
        if instructions[i].mnemonic() == Mnemonic::Call && instructions[i].is_call_near() {
            let target_address = instructions[i].near_branch_target();
            if !functions.iter().any(|f| f.start == target_address) {
                functions.push(Function {
                    start: target_address,
                    end: 0,
                });
            }
        }
        i += 1;
    }

    Ok(functions)
}
