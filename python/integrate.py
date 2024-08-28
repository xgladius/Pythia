from iced_x86 import *
from elftools.elf.elffile import ELFFile
import argparse
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import json
from typing import Dict, Sequence
from types import ModuleType
import os
import tensorflow as tf
import logging
import re

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
except ModuleNotFoundError:
    pass
tf.keras.config.disable_interactive_logging()


with open('models/tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    
model_start = load_model('models/model_start.keras')
model_end = load_model('models/model_end.keras')

def get_code_section_offset(elf_file):
    elf = ELFFile(elf_file)

    for section in elf.iter_sections():
        if section.name == '.text':
            return section['sh_offset'], section['sh_size']

    raise Exception("Code section (.text) not found in the ELF file.")

def process_operands(operands: str) -> str:
    if len(operands) <= 4:
        return re.sub(r'\b\d+\b', '', operands).strip()
    else:
        return ""

def is_function_start(instructions, formatter):
    instruction_str = " ".join([f"{mnemonic_to_string(instr.mnemonic).lower()} {process_operands(formatter.format_all_operands(instr))}" for instr in instructions])
    new_mnemonics_seq = tokenizer.texts_to_sequences([instruction_str])
    max_len = model_start.input_shape[1]
    new_mnemonics_padded = preprocessing.sequence.pad_sequences(new_mnemonics_seq, maxlen=max_len, padding='post')
    start_prediction = model_start.predict(new_mnemonics_padded)
    return start_prediction[0][0] >= 0.989000000000000, start_prediction[0][0]

def is_function_end(instructions, formatter):
    instruction_str = " ".join([f"{mnemonic_to_string(instr.mnemonic).lower()} {process_operands(formatter.format_all_operands(instr))}" for instr in instructions])
    new_mnemonics_seq = tokenizer.texts_to_sequences([instruction_str])
    max_len = model_end.input_shape[1]
    new_mnemonics_padded = preprocessing.sequence.pad_sequences(new_mnemonics_seq, maxlen=max_len, padding='post')
    end_prediction = model_end.predict(new_mnemonics_padded)
    return end_prediction[0][0] > 0.9, end_prediction[0][0]

def create_enum_dict(module: ModuleType) -> Dict[int, str]:
    return {module.__dict__[key]:key for key in module.__dict__ if isinstance(module.__dict__[key], int)}

MNEMONIC_TO_STRING: Dict[Mnemonic_, str] = create_enum_dict(Mnemonic)
def mnemonic_to_string(value: Mnemonic_) -> str:
    s = MNEMONIC_TO_STRING.get(value)
    if s is None:
        return str(value) + " /*Mnemonic enum*/"
    return s

class Function:
    def __init__(self, start = 0, end = 0):
        self.start = start
        self.end = end

    def has_end(self):
        return self.end != 0

    def has_start(self):
        return self.start != 0

def decode(code, ip):
    decoder = Decoder(64, code, ip=ip)
    formatter = Formatter(FormatterSyntax.INTEL)
    instructions = []
    functions = []
    current_function = Function()
    
    for instr in decoder:
        instructions.append(instr)
        
    i = 0
    while i < len(instructions):
        cur_candidates = instructions[i:i+4]
        is_start, start_accuracy = is_function_start(cur_candidates, formatter)
        # is_end, end_accuracy = is_function_start(instructions[i:i+4], formatter)
        
        if is_start:
            current_function.start = cur_candidates[0].ip
            print(f"found function start ({start_accuracy} confidence) at {cur_candidates[0].ip:X} for "
                f"{' '.join([f'{mnemonic_to_string(instr.mnemonic).lower()} {process_operands(formatter.format_all_operands(instr))}' for instr in cur_candidates])}")
            i += len(cur_candidates) + 1
        else:
            i += 1

def start(file_path):
    with open(file_path, 'rb') as f:
        offset, size = get_code_section_offset(f)
        f.seek(offset)
        code = f.read(size)
        ip = offset
        decode(code, ip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "get functions in elf executable")
    parser.add_argument("-f", "--file", help = "Input executable")
    args = parser.parse_args()
    start(args.file)
