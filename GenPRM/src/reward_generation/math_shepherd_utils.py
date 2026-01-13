# math_shepherd_utils.py

import argparse
import copy
import json
import os
import random
import time

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import jsonlines
import numpy as np
from transformers import AutoTokenizer

from utils.llm_utils import LLMService
from utils.grader import math_equal
from utils.parse_utils import parse_ground_truth, extract_answer

import threading

exit_code = None
# exit_code_lock = threading.Lock()


def load_data(args):
    if not args.only_monte_carlo:
        problem_answer_pairs = []
        splits = args.split.split('-')
        for s in splits:
            data_path = os.path.join(args.base_path, args.data_dir, args.data_name, f'{s}')
            data = load_from_disk(data_path)
            unique_data = {}
            for item in data:
                problem_str = str(item['problem'])
                if problem_str not in unique_data:
                    unique_data[problem_str] = item

            deduplicated_data = list(unique_data.values())

            for idx, item in tqdm(enumerate(deduplicated_data), desc=f"Processing {s}"):
                meta_data = {}
                for k, v in item.items():
                    if k not in ['problem', 'question', 'solution', 'answer', 'extracted_groundtruth']:
                        meta_data[k] = v

                problem = str(item['problem'])
                solution = str(item['solution'])
                extracted_groundtruth = str(item['answer'])

                new_item = {
                    'idx': idx,
                    'problem': problem,
                    'solution': solution,
                    'extracted_groundtruth': extracted_groundtruth,
                    'meta_data': meta_data
                }
                problem_answer_pairs.append(new_item)

        print(f"Loaded {len(problem_answer_pairs)} {args.split} problem-answer pairs.")
        return problem_answer_pairs
    else: # base_path: main_branch, data_dir: GenPRM/_data, data_name: math, split: train...
        # args.save_dir = os.path.join(args.base_path, args.save_dir, f'{args.data_name}_{args.split}_{args.model_path.split("/")[-1].split("--")[-1]}_PRM-Data')
        splits = args.split.split('-')
        for s in splits:
            if not args.special_data_dir:
                data_path = os.path.join(args.base_path, args.origin_save_dir, f'{args.data_name}_{s}_{args.origin_model_path.split("/")[-1].split("--")[-1]}_PRM-Data')
            else:
                data_path = args.data_dir
            def read_json_files_to_list(folder_path, idx=None):
                print(f"Reading JSON files from {folder_path}")
                json_files = []
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(".json"):
                            json_files.append(os.path.join(root, file))
                data = []
                for idx, json_file in tqdm(enumerate(json_files), desc="Reading JSON files"):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if 'steps' in json_data:
                            data.append(json_data)
                return data
            
            problem_answer_pairs = read_json_files_to_list(data_path, idx=args.round)
            print(f"Loaded {len(problem_answer_pairs)} {args.split} problem-answer pairs.")
            return problem_answer_pairs

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def check_correctness(full_answer: str, extracted_groundtruth: str, data_name: str) -> bool:
    if full_answer == '<INVALID>':
        return False
    # if data_name == 'math':
    #     example = {'solution': full_answer}
    # elif data_name == 'gsm8k':
    #     example = {'answer': full_answer}
    # else:
    #     raise ValueError(f"Invalid data_name: {data_name}")
    # cot, answer = parse_ground_truth(example=example, data_name=data_name)
    answer = extract_answer(full_answer, data_name)
    result = math_equal(answer, extracted_groundtruth)
    return result


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def load_model(args):
    model_service = LLMService(model_name=args.model_path, model_type="vllm", device=args.device, use_tqdm=False)
    model_service.start_service()
    tokenizer = model_service.tokenizer

    return model_service, tokenizer


def format_query(system_prompt, user_content, assistant_content):
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if user_content:
        messages.append({'role': 'user', 'content': user_content})
    if assistant_content:
        messages.append({'role': 'assistant', 'content': assistant_content})
    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # if prompt.endswith(f"{tokenizer.eos_token}\n"):
    #     prompt = prompt[:-len(f"{tokenizer.eos_token}\n")]
    # elif prompt.endswith(tokenizer.eos_token):
    #     prompt = prompt[:-len(tokenizer.eos_token)]
    return messages


def post_process_act(action, step_idx: int = 1, add_prm_step_tag: bool = False):
    if action == '<INVALID>':
        return [action]
    prm_step_tag = "ки\n"
    action = action.strip()  # Remove leading and trailing whitespaces

    if not action.startswith(f"Step "):
        action = f"Step {step_idx}: " + action
    splits = action.split("\nStep ")
    splits = [s.strip() for s in splits]
    action_list = []
    if len(splits) == 1:
        action_list.append(splits[0])
        if add_prm_step_tag:
            action_list[-1] += f" {prm_step_tag}"
    elif len(splits) > 1:
        action_list.append(splits[0])
        if add_prm_step_tag:
            action_list[-1] += f" {prm_step_tag}"
        for i in range(1, len(splits)):
            s = splits[i]
            try:
                colon_idx = s.index(":")
                s = s[:(colon_idx + 1)] + " " + s[(colon_idx + 1):].strip()
            except:
                pass
            action_list.append(f"Step {s}")
            if add_prm_step_tag:
                action_list[-1] += f" {prm_step_tag}"
    return action_list


def is_file_processed(args, file_path):
    if not is_file_exists(file_path):
        return 0
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'monte_carlo_scores' in data.keys() and len(data['monte_carlo_scores']) == args.num_paths:
                return 3
            elif 'steps' in data.keys() and len(data['steps']) == args.num_paths:
                return 2
            elif 'base_score' in data.keys():
                return 1
    except Exception as e:
        return 0
    return 0


def is_file_exists(file_path):
    return os.path.exists(file_path)


def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def create_empty_file(file_path):
    with open(file_path, 'w') as f:
        pass


def create_file_atomically(file_path):
    fd = os.open(file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        pass
        # f.write(json.dumps({'assigned_at': time.time()}))


def load_jsonl(file_path):
    data = []
    try:
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader:
                data.append(obj)
    except Exception as e:
        print(f"Load jsonl file error: {str(e)}")
    return data


def get_jsonl_file_num(save_dir, q_idx):
    question_path = os.path.join(save_dir, f"question_{q_idx}")
    if not is_file_exists(question_path):
        return 0
    file_list = [file for file in os.listdir(question_path) if file.endswith(".jsonl")]
    return len(file_list)


def get_file_exist_time(file_path):
    current_time = time.time()
    try:
        creation_time = os.path.getmtime(file_path)
    except Exception as e:
        print(f"Get file exist time error: {str(e)}")
        return 99999999
    return (current_time - creation_time) / 60.0


def get_current_save_idx():
    current_time = time.time()
    save_idx = int(round(current_time, 4) * 10000)
    return save_idx


def get_step_cnt(answer):
    counter = 1
    for i in range(2, 201):
        if f"ки\n## Step {i}: " in answer:
            counter = i
        elif f"ки\nStep {i}: " in answer:
            counter = i
        else:
            break
    return counter


def to_raw_string(string):
    encoded_string = string.encode('unicode_escape')
    raw_string = encoded_string.decode('ascii')
    return raw_string


def get_max_file_exist_time(file_path):
    max_exist_time = 0
    for file in os.listdir(file_path):
        if file.endswith(".json"):
            exist_time = get_file_exist_time(os.path.join(file_path, file))
            max_exist_time = max(max_exist_time, exist_time)
    return max_exist_time


def check_question_finished(question_path, num_sequence, save_dir, lock_dir, round_num, detailed=False):
    if not is_file_exists(question_path):
        return False, False, 0

    file_list = [file for file in os.listdir(question_path) if file.endswith(".json")]
    file_cnt = 0
    success_flag = False
    for file_name in file_list:
        file_path = os.path.join(question_path, file_name)
        q_idx = int(question_path.split('_')[-1])
        lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{q_idx}_{file_name.split('_')[-1].split('.')[0]}.lock")
        if detailed:
            try:
                obj = load_json(file_path)
                file_cnt += 1
                success_flag = True
            except Exception as e:
                print(f"Read file error: {str(e)}")
                try:
                    print(f"Remove file {file_path} due to read error")
                    os.remove(file_path)
                    if is_file_exists(lock_file_path):
                        os.remove(lock_file_path)
                except Exception as e:
                    print(f"Remove file error: {str(e)}")
        else:
            if not is_file_empty(file_path):
                file_cnt += 1
    if file_cnt >= 1:
        finished_path = os.path.join(question_path, f"finished_{round_num}.txt")
        if not os.path.exists(finished_path):
            try:
                with open(finished_path, "w") as f:
                    f.write("finished")
            except Exception as e:
                print(f"Write finished file error: {str(e)}")
        if success_flag:
            finished_path = os.path.join(question_path, f"finished.txt")
            if not os.path.exists(finished_path):
                try:
                    with open(finished_path, "w") as f:
                        f.write("finished")
                except Exception as e:
                    print(f"Write finished file error: {str(e)}")
        return success_flag, True, file_cnt
    else:
        if is_file_exists(os.path.join(question_path, f"finished_{round_num}.txt")):
            try:
                os.remove(os.path.join(question_path, f"finished_{round_num}.txt"))
            except Exception as e:
                print(f"Remove finished file error: {str(e)}")
        if is_file_exists(os.path.join(question_path, f"finished.txt")):
            try:
                os.remove(os.path.join(question_path, f"finished.txt"))
            except Exception as e:
                print(f"Remove finished file error: {str(e)}")
        return False, False, file_cnt


def check_lock_timeout(raw_test_ds, save_dir, lock_dir, round_num, max_exist_time):
    for i in range(len(raw_test_ds)):
        # if is_file_exists(os.path.join(save_dir, f"question_{i}/finished_{round_num}.txt")):
        #     continue
        lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{round_num}.lock")
        if is_file_exists(lock_file_path):
            exist_time = get_file_exist_time(lock_file_path)
            if exist_time > max_exist_time:
                try:
                    os.remove(lock_file_path)
                    print(f"Remove lock file {lock_file_path}, exist time: {exist_time:.1f} minutes")
                except Exception as e:
                    print(f"Remove lock file error: {str(e)}")


def check_process_cnt(raw_test_ds, save_dir):
    total_cnt = 0
    for i in range(len(raw_test_ds)):
        file_cnt = 0
        question_path = os.path.join(save_dir, f"question_{i}")
        if not is_file_exists(question_path):
            continue
        file_list = [file for file in os.listdir(question_path) if file.endswith(".json")]
        for file_name in file_list:
            if not is_file_empty(os.path.join(question_path, file_name)):
                file_cnt += 1
            if file_cnt >= 1:
                break
        total_cnt += file_cnt

    return total_cnt


def assign_tasks(raw_test_ds, question_mask, num_sequence, save_dir, lock_dir, round_num=0, eager=False, batch_size=0, max_exist_time=0):
    """
    assign_tasks for current worker
    """
    global exit_code
    check_lock_timeout(raw_test_ds, save_dir, lock_dir, round_num, max_exist_time)

    success_flags = [False for _ in range(len(raw_test_ds))]
    for i in range(len(raw_test_ds)):
        question_path = os.path.join(save_dir, f"question_{i}")
        success_flag, flag, file_cnt = check_question_finished(question_path, num_sequence, save_dir, lock_dir, round_num, detailed=True)
        success_flags[i] = success_flag

    print(f'Round: {round_num}, Batch size: {batch_size}, Max exist time: {max_exist_time}')

    start_time = time.time()
    print(f'Assign start at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

    test_ds = []
    chosen_idxs = []
    chosen_dict = {}

    j = round_num
    shuffled_indices = list(range(len(raw_test_ds)))
    random.shuffle(shuffled_indices)

    for i in shuffled_indices:
        if question_mask[i] and not success_flags[i]:
            file_path = os.path.join(save_dir, f"question_{i}/record_{j}.json")
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")

            if is_file_exists(file_path):
                if is_file_empty(file_path):
                    exist_time = get_file_exist_time(file_path)
                    try:
                        print(f"Remove empty file {file_path}, exist time: {exist_time:.1f} minutes")
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Remove empty file error: {str(e)}")
                    if is_file_exists(lock_file_path):
                        try:
                            print(f"Remove lock file {lock_file_path} at the same time")
                            os.remove(lock_file_path)
                        except Exception as e:
                            print(f"Remove lock file error: {str(e)}")
                    chosen_idxs.append([i, j])
                    if i not in chosen_dict.keys():
                        chosen_dict[i] = [j]
                    else:
                        chosen_dict[i].append(j)
                    test_ds.append(raw_test_ds[i])
                    create_empty_file(lock_file_path)
                else:
                    continue
            else:
                if eager:
                    chosen_idxs.append([i, j])
                    if i not in chosen_dict.keys():
                        chosen_dict[i] = [j]
                    else:
                        chosen_dict[i].append(j)
                    test_ds.append(raw_test_ds[i])
                    create_empty_file(lock_file_path)
                else:
                    if is_file_exists(lock_file_path):
                        exist_time = get_file_exist_time(lock_file_path)
                        if exist_time > max_exist_time:
                            try:
                                print(f"Remove lock file {lock_file_path}, exist time: {exist_time:.1f} minutes")
                                os.remove(lock_file_path)
                                chosen_idxs.append([i, j])
                                if i not in chosen_dict.keys():
                                    chosen_dict[i] = [j]
                                else:
                                    chosen_dict[i].append(j)
                                test_ds.append(raw_test_ds[i])
                                create_empty_file(lock_file_path)
                            except Exception as e:
                                print(f"Remove lock file error: {str(e)}")
                                continue
                        else:
                            continue
                    else:
                        chosen_idxs.append([i, j])
                        if i not in chosen_dict.keys():
                            chosen_dict[i] = [j]
                        else:
                            chosen_dict[i].append(j)
                        test_ds.append(raw_test_ds[i])
                        create_empty_file(lock_file_path)
            if len(chosen_idxs) >= batch_size:
                break
        if len(chosen_idxs) >= batch_size:
            break

    print(f"Len: {len(chosen_idxs)}, Chosen idxs: {chosen_idxs}")
    print(f"Assign end at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, Time cost: {time.time() - start_time:.1f} seconds")

    # refresh_thread = threading.Thread(target=heart_beat_worker, args=(chosen_idxs, save_dir, lock_dir), daemon=True)
    # refresh_thread.start()

    print(f"Len: {len(test_ds)}, Chosen dict: {chosen_dict}")
    return test_ds, chosen_dict, chosen_idxs, save_dir, lock_dir
