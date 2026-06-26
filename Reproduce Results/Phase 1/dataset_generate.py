import os
import json
import re
import argparse
import math
from itertools import islice
from decimal import Decimal, getcontext

def generate_state_probabilities(marked_states, n_qubits, p_marked_total, t):
    state_dict = {}
    total_states = 2 ** n_qubits
    
    per_marked = p_marked_total/t
    
    for marked_state in marked_states:
        state_dict[marked_state] = round(per_marked, 4)
    
    unmarked_total = 1 - p_marked_total
    unmarked_count = total_states - t
    
    if unmarked_count > 0:
        per_unmarked_raw = unmarked_total / unmarked_count
        rounded = round(per_unmarked_raw, 4)
        if rounded == 0.0 and per_unmarked_raw != 0:
            per_unmarked = "{:.4e}".format(per_unmarked_raw)  
        else:
            per_unmarked = rounded
    else:
        per_unmarked = 0.0

    max_display = 30 - t  
    count = 0
    
    for i in range(total_states):
        state = format(i, '0{}b'.format(n_qubits))
        if state in marked_states:
            continue
        if count < max_display:
            state_dict[state] = per_unmarked
            count += 1
        else:
            break
    
    return state_dict

def extract_oracle_structure(qasm_content):
    """Extract Oracle structure, return parameter list and operation sequence"""
    oracle_match = re.search(
        r'gate Oracle\s*((?:[\w]+,?\s*)+)\s*{([^}]*)}',
        qasm_content,
        re.DOTALL
    )
    if not oracle_match:
        return [], []
    
    oracle_head = "gate Oracle " + oracle_match.group(1)
    
    # Parameter list maintains original declaration order (Qiskit MSB → LSB)
    params = [p.strip() for p in oracle_match.group(1).split(',')]
    operations = [
        op.strip() 
        for op in oracle_match.group(2).split(';') 
        if op.strip()
    ]
    return params, operations, oracle_head, oracle_match.group(2)

def analyze_marked_state(params, operations):
    mcmt_indices = [i for i, op in enumerate(operations) if op.startswith('mcmt')]
    marked_states = []
    
    blocks = []
    prev_end = 0
    for mcmt_idx in mcmt_indices:
        start = find_block_start(operations, mcmt_idx, prev_end)
        end = find_block_end(operations, mcmt_idx, start)
        blocks.append((start, mcmt_idx, end))
        prev_end = end + 1
    
    x_ops_before_list = []
    
    for start, mcmt_idx, end in blocks:
        # Extract X operations before mcmt
        pre_mcmt_ops = operations[start: mcmt_idx]
        x_ops_before = [op for op in pre_mcmt_ops if op.startswith('x ')]
        x_ops_before_list.append(x_ops_before)
        
        current_state = ''
        block_ops = operations[start:end+1]
        
        for q in params: 
            pre_x = any(op == f'x {q}' for op in block_ops[:block_ops.index(f'mcmt {", ".join(params)}')])
            post_x = any(op == f'x {q}' for op in block_ops[block_ops.index(f'mcmt {", ".join(params)}')+1:])
            bit = '0' if (pre_x and post_x) else '1'
            current_state = bit + current_state
        
        marked_states.append(current_state)
    return marked_states, blocks, x_ops_before_list

def compute_theta(t, n):
    """
    Calculate the angle θ for Grover's algorithm
    :param t: Number of marked states
    :param n: Number of qubits
    :return: angle θ
    """
    N = 2 ** n
    if t < 0 or t > N:
        raise ValueError("t must be between 0 and N (inclusive)")
    ratio = t / N
    sqrt_ratio = math.sqrt(ratio)
    theta = math.asin(sqrt_ratio)
    return theta

def grover_probabilities(t, n, k):
    """
    Calculate the probabilities of marked and unmarked states
    :param t: Number of marked states
    :param n: Number of qubits
    :return: ( p_marked, p_unmarked )
    """
    theta = compute_theta(t, n)
    angle = (2 * k + 1) * theta
    p_marked = math.sin(angle) ** 2
    p_unmarked = math.cos(angle) ** 2
    
    return p_marked, p_unmarked

def generate_reasoning(params, operations, marked_states, blocks, x_ops_before_list, oracle_head, oracle_body): 

    analysis = [
        "=== Analysis ===",
        f"", 
        f"The Oracle entity is extracted below:", 
        f"{oracle_body}", 
    ]
    
    for block_idx, (start, mcmt_idx, end) in enumerate(blocks):
        
        block_ops = operations[start:end+1]
        code_snippet = []
        
        for op in block_ops:
            code_snippet.append(f"{op};" if op.startswith('mcmt') else f"{op};")

        state_steps = []
        current_state = ''
        for q in params:
            pre_x = any(op == f'x {q}' for op in block_ops[:block_ops.index(f'mcmt {", ".join(params)}')])
            post_x = any(op == f'x {q}' for op in block_ops[block_ops.index(f'mcmt {", ".join(params)}')+1:])
            bit = '0' if (pre_x and post_x) else '1'
            current_state = bit + current_state  
            state_steps.append(f"x {q}: {'Present' if (pre_x and post_x) else 'Absent'} → {bit}, then → {current_state}")
        
        analysis.extend([
            f"\n=== Block {block_idx+1} ===",
            "Operation sequence:",
            *code_snippet,
        ])

        analysis.extend([
            "\nState construction:",
            *[f"{i+1}. {step}" for i, step in enumerate(state_steps)],
            f"Final state: {current_state}",
        ])
    
    analysis.append("\n=== Final Marked States ===")
    analysis.extend(marked_states)

    # Calculate probabilities
    n_qubits = len(params)
    t = len(marked_states)  
    N = 2 ** n_qubits
    
    theta = compute_theta(t, n_qubits)
    k_opt =  math.floor(math.pi / 4 * math.sqrt((2 ** n_qubits) / t))
    p_marked, _ = grover_probabilities(t, n_qubits, k_opt)
    
    state_probs = generate_state_probabilities(marked_states, n_qubits, p_marked, t)
    
    prob_output = [
        "\n=== Simulation Results of the Grover's Algorithm ==="
        "\n{" + 
        ",\n ".join([f"'{k}': {v}" if isinstance(v, str) else f"'{k}': {v:.4f}" for k, v in state_probs.items()]) +
        "}"
    ]

    return '\n'.join(analysis + prob_output)

def find_block_start(ops, mcmt_idx, prev_end):
    start = mcmt_idx
    while start > prev_end:
        if ops[start-1].startswith('x '):
            start -= 1
        else:
            break
    return start

def find_block_end(ops, mcmt_idx, start):
    return mcmt_idx + (mcmt_idx - start)

def extract_oracle_gate(qasm_content):
    """Extract the Oracle gate definition from QASM content"""
    lines = qasm_content.split('\n')
    oracle_block = []
    in_oracle = False
    brace_count = 0
    
    for line in lines:
        # Detect Oracle gate declaration
        if line.strip().startswith('gate Oracle'):
            in_oracle = True
            brace_count = 0
        
        if in_oracle:
            oracle_block.append(line)
            brace_count += line.count('{')
            brace_count -= line.count('}')
            
            # End of Oracle gate definition
            if brace_count == 0 and line.strip().endswith('}'):
                in_oracle = False
                return '\n'.join(oracle_block)
    
    return None  # If not found

def process_qasm_files(data_dir, input_type, n_min, n_max):
    dataset = []
    no_x_dataset = []
    count = 0

    subdirs = []
    for subdir, _, files in os.walk(data_dir):
        folder_name = os.path.basename(subdir)
        if folder_name.startswith("grover_n") and folder_name[8:].isdigit():
            n_qubits = int(folder_name[8:])
            if n_min <= n_qubits <= n_max:
                subdirs.append((n_qubits, subdir, files))
    
    subdirs.sort(key=lambda x: x[0])
    
    for n_qubits, subdir, files in subdirs:
        for file in files:
            if file.endswith(".qasm"):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                params, ops, oracle_head, oracle_body = extract_oracle_structure(content)
                oracles = extract_oracle_gate(content)
                if not params:
                    continue
                
                marked_state, blocks, x_ops_before_list = analyze_marked_state(params, ops)
                reasoning = generate_reasoning(params, ops, marked_state, blocks, x_ops_before_list, oracle_head, oracle_body)
                
                entry = {
                    "instruction": f"",
                    "input": oracles if input_type == "Oracle" else content,
                    "output": reasoning
                }
                
                dataset.append(entry)
                
                if "No existed x _gate_q_ before mcmt." in entry["output"]:
                    no_x_dataset.append(entry)
    
    # Save dataset
    with open(f'Grover_{input_type}_{n_min}_{n_max}_MMS.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process QASM files in the specified directory.")
    parser.add_argument("--directory", type=str, default="data_MMS", help="The directory containing QASM files.")
    parser.add_argument("--input_type", type=str, choices=["Oracle", "FullCircuit"], default="FullCircuit",
                        help="Specify whether to use 'Oracle' or 'FullCircuit' as the input field.")
    parser.add_argument("--n_min", type=int, default=2, help="Minimum n_qubits to process.")
    parser.add_argument("--n_max", type=int, default=7, help="Maximum n_qubits to process.")
    args = parser.parse_args()
    process_qasm_files(args.directory, args.input_type, args.n_min, args.n_max)