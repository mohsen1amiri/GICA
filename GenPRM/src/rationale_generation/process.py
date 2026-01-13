import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import argparse
import json
import random
import time
import copy
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_utils import LLMService
from utils.util import *
import threading
from colorama import Fore, init
import io
from contextlib import redirect_stdout
import signal

os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_USE_V1'] = '1'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

stop_event = threading.Event()

BOXED_PATTERN1 = re.compile(
    r'(\\\[\s*)?\\boxed\s*{\s*(\\text\s*{\s*)?([yY][eE][sS]|[nN][oO])\s*(}\s*)?}(\s*\\\])?',
    flags=re.IGNORECASE
)

BOXED_PATTERN2 = re.compile(
    r'(\\\[\s*)?\\boxed\s*{\s*(\\text\s*{\s*)?((?:[^{}]|\{[^{}]*\})*)\s*(}\s*)?}(\s*\\\])?',
    flags=re.IGNORECASE
)

def normalize_boxed(content, must_yes_no=False):
    """
    normalize the boxed{} and return the boxed content
    """
    results = []
    flags = []

    def repl(match):
        inner_content = match.group(3).strip()
        normalized_val = inner_content
        current_flag = None
        
        if inner_content.lower() in {'yes', 'no'}:
            normalized_val = inner_content.capitalize()
            current_flag = 1 if normalized_val == 'Yes' else 0
        
        new_boxed = f'\\boxed{{{normalized_val}}}'
        results.append(normalized_val)
        
        if current_flag is not None:
            flags.append(current_flag)
        
        return new_boxed

    if must_yes_no:
        new_content, found = BOXED_PATTERN1.subn(repl, content)
    else:
        new_content, found = BOXED_PATTERN2.subn(repl, content)
    
    if not found:
        return content, None, None, None, None
    
    last_result = results[-1] if results else None
    last_flag = flags[-1] if flags else None
    
    return new_content, last_result, last_flag, results, flags

class timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
    
    def handle_timeout(self, signum, frame):
        raise TimeoutError("Code execution timed out")

class CodeExecutor:
    def __init__(self):
        self.namespace = {}
        self.code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)

    def execute(self, text):
        code_block = self.code_pattern.findall(text)[-1].strip()
        
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with timeout(seconds=5):
                    exec(code_block, self.namespace)
            actual = f.getvalue().strip()
        except TimeoutError as te:
            actual = f"Code execute time out: {te}"
            print(actual)
        except Exception as e:
            actual = f"Code execute Error: {type(e).__name__}: {e}"
            print(actual)
        
        return actual

def contains_expected_answer(target_string):
    if re.search(r"reference solution", target_string, re.IGNORECASE):
        return True
    elif re.search(r"expected answer", target_string, re.IGNORECASE):
        return True
    else:
        return False

def get_json_files(data_path):
    random.seed(int(time.time()))
    files = []
    
    base_path = os.path.normpath(data_path)
    
    for root, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, start=base_path)
                files.append(rel_path)
    
    random.shuffle(files)
    return files

def check_assistant_response(assistant_response):
    if assistant_response is None:
        return False
    
    # if contains_expected_answer(assistant_response):
    #     return False
    
    if '### Paragraph' not in assistant_response:
        return False
    
    # if 'boxed{No}' not in assistant_response and 'boxed{Yes}' not in assistant_response:
    #     return False
    
    # if '### Final Answer' not in assistant_response:
    #     return False
    
    # def has_boxed_integer(text):
    #     pattern = r'boxed\{-?\d+\}'
    #     return re.search(pattern, text) is not None
    
    # if not has_boxed_integer(assistant_response):
    #     return False

    return True

def get_majority_vote_responses(args, model, messages, tokenizer, assistant_response, flag, must_yes_no=False):
    majority_num = 1
    if not args.majority_of_N:
        majority_num = 16
    else:
        majority_num = args.majority_of_N
    
    timestamped_print(f'start majority of {majority_num}')
    first_prompt = build_prompt(messages, tokenizer)
    first_conversation = messages
    cprint(first_prompt, 'first prompt')
    timestamped_print(f'majority_num: {majority_num}')
    max_code_attempts=64
    current_conversations = [copy.deepcopy(messages) for _ in range(majority_num)]

    def contains_python_code(text):
        return bool(re.search(r'```python\s*.*?\s*```', text, re.DOTALL))

    def execute_code_loop(current_conversation, executor, depth=0):
        if depth > max_code_attempts:
            return []
        current_prompt = build_prompt(current_conversation, tokenizer)
        responses = model.generate_response(
            prompt=current_prompt, n=1, temperature=0.6, top_p=0.95, top_k=40, stop=['\n```\n', '</think>'], include_stop_str_in_output=True
        )[0]
        for i, res in enumerate(responses):
            res, _, _, _, _ = normalize_boxed(res, must_yes_no=must_yes_no)
            current_conversation[-1]['content'] += res
            if contains_python_code(res):
                print('1', end=' ')
                code_output = executor.execute(res)
                code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                current_conversation[-1]['content'] += code_content
                collected, cot_indices = execute_code_loop(current_conversation, executor, depth+1)
            elif res.endswith('</think>'):
                current_conversation[-1]['content'] = '</verify>'.join(current_conversation[-1]['content'].split('</verify>')[:-1])
                current_conversation[-1]['content'] += '</verify>\n### Final Conclusion\n\nAccording to the requirement, I need to return the **index of the paragraph where the earliest error occurs**. Otherwise, return the **index of -1 (which typically denotes "not found")**.\n\nANSWER: \\boxed'
                current_prompt = build_prompt(current_conversation, tokenizer)
                outputs = [model.generate_response(
                    prompt=current_prompt, n=1, temperature=0.6, top_p=0.95, top_k=40, stop=['}'], include_stop_str_in_output=True
                )[0] for _ in range(4)]
                responses = [out[0] for out in outputs]
                print('2', end=' ')
                cot_indices = responses
                collected = current_conversation
            else:
                current_conversation[-1]['content'] = current_conversation[-1]['content'].rstrip('\n```\n')
                collected, cot_indices = execute_code_loop(current_conversation, executor, depth+1)
                print('3', end=' ')
        return collected, cot_indices
    
    pot_outputs = [
        execute_code_loop(conv, CodeExecutor(), 0) 
        for conv in current_conversations
    ]
    final_conversations = [
        pot_outputs[i][0] for i in range(len(pot_outputs))
    ]
    cot_indices_list = [
        pot_outputs[i][1] for i in range(len(pot_outputs))
    ]

    return final_conversations, cot_indices_list, flag

def generate_critic_prompt(problem: str, steps: list, ref=None) -> str:
    # Generate solution section with step tags
    solution_steps = []
    for idx, step in enumerate(steps):
        content = re.sub(r'^Step \d+:\s*', '', step)
        solution_steps.append(
            f"<paragraph_{idx+1}>\n{content}\n</paragraph_{idx+1}>"
        )
    
    solution_section = "\n\n".join(solution_steps)
    
    # Build full prompt template
    if ref is None:
        prompt_template = f'The following is the math problem and a solution (split into paragraphs, enclosed with tags and indexed from 1):\n[Math Problem]\n\n{problem}\n\n[Solution]\n\n{solution_section}\n\nYour task is to verify the correctness of paragraph in the solution.  Split your verification by `### Paragraph {{ID}}`. \n\nYour verification for each paragraph should be constructed by 2 parts, wrapped by `<analyze></analyze>` and `<verify></verify>` separately.\n\n1. in `<analyze>` part, you need to analyze the reasoning process and explain why the paragraph is correct or incorrect in detail. \n2. in `<verify>` part, you must write **Python code** in the form of ```python\n{{CODE}}\n``` to verify every details that can be verified by code. You can import PyPI (i.e., `sympy`, `scipy` and so on) to implement complicated calculation. Make sure to print the critic results in the code. Every code will be executed automatically by system. You need to analyze the `[Code Output]` after code executing.\n\n> Pay attention that you must follow the format of ```python\n{{CODE}}\n``` when you write the code, otherwise the code will not be executed.\n\nAfter all verifications, if you identify an error in a paragraph, return the **index of the paragraph where the earliest error occurs**. Otherwise, return the **index of -1 (which typically denotes "not found")**. Please put your final answer (i.e., the index) within box in the form of `$\\boxed{{INDEX}}$`.'
    else:
        raise ValueError('no ref mode')
    return prompt_template

def process_json_file(args, filename, data_dir, conversation_dir, model, tokenizer):
    data_path = os.path.join(data_dir, filename)
    conversation_path = os.path.join(conversation_dir, filename)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        timestamped_print(f"WARNING: fail to load {data_path} Error: {e}", 'ERROR')
        return

    ref_solution = data['solution']
    problem = data.get('problem')
    steps_groups = data.get('steps')
    if not problem or not steps_groups:
        timestamped_print(f"WARNING: file {filename} is lack of 'problem' or 'steps' field", 'WARNING')
        return
    
    cot_rationale = []
    cot_indices_lists = []
    cot_flags = []
    
    for group in steps_groups:
        if not isinstance(group, list):
            timestamped_print(f"WARNING: 'steps' in file {filename} is not list", 'WARNING')
            continue
        if len(group) == 0:
            continue
        if args.expected_answer:
            raise ValueError('no ref mode')
        else:
            user_prompt = generate_critic_prompt(problem, group)
            sys_prompt = 'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step with python code.'
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': "<think>\n### Paragraph 1\n<analyze>\n"},
            ]
        if not args.majority_of_N:
            raise ValueError('must set majority_of_N')
        else:
            assistant_response = None
        
        flag = False
        
        final_conv, cot_indices_list, flag = get_majority_vote_responses(args, model, messages, tokenizer, assistant_response, flag, must_yes_no=False)
        
        messages = final_conv
        cprint(messages, 'messages')
            
        cot_flags.append(flag)
        
        cot_rationale.append(messages)
        cot_indices_lists.append(cot_indices_list)

    data['cot_rationale'] = cot_rationale
    data['cot_flags'] = cot_flags
    data['cot_indices'] = cot_indices_lists

    try:
        with open(conversation_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        timestamped_print(f"Error when saving {conversation_path}. Information: {e}", 'ERROR')

def heart_beat_worker(file_path):
    while not stop_event.is_set():
        if os.path.exists(file_path):
            try:
                os.utime(file_path)
                timestamped_print(f"Heartbeat updated: {file_path}")
            except Exception as e:
                timestamped_print(f"Update file time error: {str(e)}", 'ERROR')
        else:
            try:
                with open(file_path, 'w') as f:
                    pass
                timestamped_print(f"Created file while heart beating: {file_path}", 'ERROR')
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')
        for _ in range(6):  # 6 * 5 = 30 seconds total
            if stop_event.is_set():
                print('==============heartbeat worker exit==============')
                return
            time.sleep(5)
    print('==============heartbeat worker exit==============')

def get_file_exist_time(file_path):
    current_time = time.time()
    try:
        creation_time = os.path.getmtime(file_path)
    except Exception as e:
        timestamped_print(f"Get file exist time error: {str(e)}", 'ERROR')
        return 99999999
    return (current_time - creation_time) / 60.0

def should_process_file(file_path, timeout=15):
    try:
        if not os.path.exists(file_path):
            return True
        
        if get_file_exist_time(file_path) < timeout:
            return False
        
        if os.path.getsize(file_path) == 0:
            return True
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if 'cot_rationale' not in data:
                return True
            
            cot_rationale = data['cot_rationale']
            if not isinstance(cot_rationale, list):
                return True
            if len(cot_rationale) == 0:
                return True
            if all(not item for item in cot_rationale):
                return True
            
        return False
    
    except json.JSONDecodeError:
        return True
    except Exception as e:
        timestamped_print(f"Exception of {file_path}: {str(e)}", 'ERROR')
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate rationales with LLM.")
    parser.add_argument('--model_path', type=str, required=True, help='Model path.')
    parser.add_argument('--data_path', type=str, help='Dataset path')
    parser.add_argument('--save_path', type=str, help='Path for saving generated conversations')
    # parser.add_argument('--expected_answer', action='store_true', help='Use expected answer guide or not.')
    parser.add_argument('--num_gpu_per', type=int, default=1, help='The number of GPU to use')
    parser.add_argument('--majority_of_N', type=int, help='The number of N for majority vote')
    # parser.add_argument('--step', action='store_true', help='Use step by step mode or not.')
    return parser.parse_args()


def main():
    args = parse_args()
    print_args(args, 
              program_name="rationale_generation",
              version="1.0.0")
    
    model = LLMService(
        model_name=args.model_path,
        model_type="vllm",
        device="cuda",
        tensor_parallel_size=args.num_gpu_per,
        gpu_memory_utilization=0.95,
        dtype="auto"
    )
    model.start_service()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data_dir = args.data_path
    conversation_dir = args.save_path
    os.makedirs(conversation_dir, exist_ok=True)
    
    files = get_json_files(data_dir)
    
    timestamped_print(f"Start processing {len(files)} files...")
    for filename in tqdm(files, desc="Processing files"):
        conversation_path = os.path.join(conversation_dir, filename)
        
        if should_process_file(conversation_path, timeout=1):
            if os.path.exists(conversation_path):
                try:
                    os.remove(conversation_path)
                    timestamped_print(f"Clear invalid file: {conversation_path}")
                except Exception as e:
                    timestamped_print(f"Fail to delete file: {conversation_path}, Error: {str(e)}", 'ERROR')
                    continue
            
            try:
                os.makedirs(os.path.dirname(conversation_path), exist_ok=True)
                with open(conversation_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
            except Exception as e:
                timestamped_print(f"Fail to create file: {conversation_path}, Error: {str(e)}", 'ERROR')
                continue
            
            stop_event.clear()
            thread = threading.Thread(target=heart_beat_worker, args=(conversation_path,))
            thread.daemon = True
            thread.start()
            timestamped_print("Heartbeat thread started. Main thread continues...")

            # generate 'cot_rationale'
            try:
                process_json_file(args, filename, data_dir, conversation_dir, model, tokenizer)
            except Exception as e:
                timestamped_print(f"Error when process {filename}: {str(e)}", 'ERROR')
                import traceback
                error_info = traceback.format_exc()
                timestamped_print(f"Error when process {filename}: {str(e)}", 'ERROR')
                timestamped_print(f"Information:\n{error_info}", 'ERROR')
                exit(1)

            stop_event.set()
            thread.join(timeout=5)

            if thread.is_alive():
                timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
            else:
                timestamped_print("Heartbeat thread exited successfully")
        else:
            timestamped_print(f"Skip file: {filename}, it has been stored in {conversation_dir}")
            continue
    
    timestamped_print("Finished processing all files.")

if __name__ == "__main__":
    main()
