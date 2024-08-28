import os
import json
import glob
import random
from multiprocessing import Pool, cpu_count
import re

samples_dir = "samples"

# Function to process a single file
def process_file(filename):
    if not filename.endswith(".json"):
        json_filename = os.path.join(samples_dir, f"{filename}.json")

        # Check if the JSON file already exists
        if os.path.exists(json_filename):
            return f"Skipping {filename}, already processed."

        full_path = os.path.join(samples_dir, filename)

        print(f"Starting extraction for {filename}")

        from headless_ida import HeadlessIda
        try:
            headlessida = HeadlessIda("/home/bee/idapro-9.0/idat64", full_path)

            import idautils
            import ida_name
            import ida_funcs
            import idc
            import idaapi

            functions_data = []
            non_function_ranges = []

            # Function to process operands by removing numeric values
            def process_operands(operands):
                return [re.sub(r'\b\d+\b', '', operand).strip() for operand in operands if len(operand) <= 4]

            # Function to extract detailed instruction information
            def extract_instruction_info(start_addr, end_addr, max_instructions):
                instructions = []
                for insn in idautils.Heads(start_addr, end_addr):
                    if len(instructions) >= max_instructions:
                        break
                    mnem = idc.print_insn_mnem(insn)
                    operands = process_operands([idc.print_operand(insn, i) for i in range(3)])
                    if mnem:
                        instructions.append({
                            "mnemonic": mnem,
                            "operands": operands,
                        })
                return instructions

            # Collect function boundaries and their detailed instruction information
            for func in idautils.Functions():
                func_start = func
                func_t = ida_funcs.get_func(func)
                func_end = func_t.end_ea

                # Fetch detailed instructions for prologue and epilogue
                prologue_info = extract_instruction_info(func_start, func_start + 16, max_instructions=5)
                epilogue_info = extract_instruction_info(func_end - 16, func_end, max_instructions=5)

                # Add function data to the list
                functions_data.append({
                    "start_address": hex(func_start),
                    "end_address": hex(func_end),
                    "function_length": func_end - func_start,
                    "prologue_info": prologue_info,
                    "epilogue_info": epilogue_info
                })

                # Mark this range as used by a function
                non_function_ranges.append((func_start, func_end))

            # Collect negative examples (non-function areas)
            binary_size = idc.get_segm_end(idc.get_first_seg())
            negative_examples = []

            for func in functions_data:
                func_start = int(func["start_address"], 16)
                func_end = int(func["end_address"], 16)
                func_size = func_end - func_start
                prologue_info = func["prologue_info"]
                epilogue_info = func["epilogue_info"]

                # Only consider negative samples from within the function body, not prologue/epilogue
                if func_size > 32:  # Ensure there's enough space for the body excluding prologue and epilogue
                    body_start = func_start + 16
                    body_end = func_end - 16
                    potential_negatives = []

                    # Scan the function body for instruction sequences
                    for addr in idautils.Heads(body_start, body_end):
                        if addr >= body_end - 5 * 4:  # To ensure we have enough instructions
                            break
                        candidate_info = extract_instruction_info(addr, addr + 16, max_instructions=5)

                        if candidate_info:
                            # Check if candidate shares at least one mnemonic with prologue or epilogue
                            candidate_mnemonics = [insn['mnemonic'] for insn in candidate_info]
                            if any(m in [insn['mnemonic'] for insn in prologue_info] or
                                   m in [insn['mnemonic'] for insn in epilogue_info] for m in candidate_mnemonics):
                                potential_negatives.append((addr, candidate_info))

                    # Randomly select one negative example from potential candidates
                    if potential_negatives:
                        selected_addr, selected_info = random.choice(potential_negatives)
                        negative_examples.append({
                            "start_address": hex(selected_addr),
                            "end_address": hex(selected_addr + 16),
                            "instructions": selected_info,
                            "label_start": 0,
                            "label_end": 0
                        })

            # Save both function data and negative examples
            with open(json_filename, 'w') as json_file:
                json.dump({
                    "functions": functions_data,
                    "negative_examples": negative_examples
                }, json_file, indent=4)

            print(f"Function and negative example data for {filename} written to {json_filename}")
        except Exception as e:
            return f"Failed for file {full_path}: {e}"
        directory = "/tmp/"
        extensions = ["*.til", "*.nam", "*.id0", "*.id1", "*.id2"]
        for ext in extensions:
            pattern = os.path.join(directory, ext)
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")

# Main entry point
if __name__ == '__main__':
    # List all files in the samples directory
    files = os.listdir(samples_dir)

    # Use a pool of workers to process files in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, files)
