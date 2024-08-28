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

#[derive(Debug)]
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
            predictor.is_function_start(cur_candidates, &mut formatter, 0.999); // Super confident in the start model

        if is_start {
            current_function.start = cur_candidates[0].ip();
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

    // Make two assumptions,
    // 1. That we have found all function prologues
    // 2. That functions are contiguous in memory
    // This allows us to walk backwards in our function vec to find the end of a function
    // functions[i].end = functions[i+1].start - sizeof(last valid instruction found before functions[i+1].start) && predicted_end == true
    for i in 0..functions.len() - 1 {
        let next_function_start = functions[i + 1].start;

        let mut function_end = 0;
        for j in (0..instructions.len()).rev() {
            let instruction = instructions[j];
            if instruction.ip() < next_function_start && instructions[i].mnemonic() != Mnemonic::Nop
            {
                let (is_end, confidence) =
                    predictor.is_function_end(&instructions[j..j + 4], &mut formatter, 0.98); // Not as confident in the end model
                if is_end {
                    function_end = instruction.ip();
                    break;
                }
            }
        }

        functions[i].end = function_end;
    }

    Ok(functions)
}
