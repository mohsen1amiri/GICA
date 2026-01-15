
import os
import sys
import json
from statistics import mean
import numpy as np
import re
import regex
from text_utils import strip_string

sys.path.append('..')
from prm_evaluation.genprm_inference import GenPRM
from prm_evaluation.genprm_inference import CodeExecutor
genprm = GenPRM('GenPRM/GenPRM-1.5B', tensor_parallel_size=4)

messages = [ 
	{ "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.", "role": "system" }, 
	{ "role": "user", "content": "Question: What numeral is in the 100th decimal place in the decimal representation of $\\frac{6}{7}$?\n\nTo find the numeral in the 100th decimal place of $\\frac{6}{7}$, we first need to determine the decimal representation of $\\frac{6}{7}$." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "Performing the division, we find that $\\frac{6}{7} = 0.\\overline{857142}$, which means the decimal representation is a repeating decimal with a period of 6." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "The repeating block is \"857142\", and it repeats indefinitely." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "To find the numeral in the 100th decimal place, we need to determine the position of the 100th decimal within this repeating block." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "We do this by finding the remainder when 100 is divided by the length of the repeating block, which is 6." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "Calculating the remainder, we get $100 \\div 6 = 16$ with a remainder of 4. This means the 100th decimal place corresponds to the 4th digit in the repeating block \"857142\"." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "Therefore, the numeral in the 100th decimal place is the 4th digit of the repeating block, which is $4$." }, 
	{ "role": 'assistant', 'content': ''},
	{ "role": "user", "content": "The remainder of 4 indicates that the 100th decimal place is the same as the 4th digit in the repeating block, which is 10.\n\nAnswer: \\boxed{10}" },
	{ "role": 'assistant', 'content': ''},
]

code_executor = CodeExecutor()
reward_list = []

with open("data/Deepseek-Math-RL-7B.json") as f:
	data = json.load(f)	
exact_match =0
mismatches=0
for idx in range(len(data["answer"])):
	prompt = data["prompt"][idx]
	path_scores = []
	for index, path in enumerate(data["completion"][idx]):
		messages = [{ "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.", "role": "system" }, 
]
		steps = path.split(".\n")
		messages.append({ "role": "user", "content": "Question: " + prompt + "\n\n" + steps[0] })
		messages.append({ "role": 'assistant', 'content': ''})
		for step in steps[1:]:
			messages.append({ "role": "user", "content": step })
			messages.append({ "role": 'assistant', 'content': ''})		
	
		for i in range(len(messages)):
			if messages[i]['role'] != 'assistant':
				continue
			try:
				output, reward = genprm.inference(messages[:i], cur_step=int(i), code_executor=code_executor, logging=False)
			except:
				continue
			messages[i]['content'] = output[0]
			reward_list.append(reward)


		print("index",idx, index)
		path_scores.append(mean(reward_list))
	max_idx = np.argmax(np.array(path_scores))
	winning_path = data["completion"][idx][max_idx].lower()
	winning_path = winning_path.replace("\n", "")
	winning_path = winning_path.lower().split("the answer is:")[-1]
	winning_path = winning_path.replace("$","")
	winning_path = strip_string(winning_path)
	ground_truth = strip_string(data["answer"][idx])
	print("winning_path**",winning_path, ground_truth.strip().lower())
	if winning_path.strip().lower() == ground_truth.strip().lower():
		exact_match+=1
	else:
		mismatches+=1
	print("EM so far", exact_match/(exact_match+mismatches))
print("Final EM", exact_match/(exact_match+mismatches))
