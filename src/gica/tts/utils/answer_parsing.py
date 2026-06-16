import re
from typing import Optional
from collections import Counter
import re



def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except Exception as e:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except Exception as e:
        pass
    return output == answer


def get_majoirty_answer(answers):
    if len(answers) == 0: # all solutions are invalid -- do not have a final answer
        return '[INVALID]'
    else:
        ## pick the majority answer
        voted_answer = Counter(answers).most_common(1)[0][0]
        ## pick the first solution that has the voted answer
        return voted_answer
    

def get_action_trace_from_plan_str(plan_str):
    ### return all lines after lines with [ACTION] 
    lines = plan_str.split('\n')
    action_trace = []
    for i, line in enumerate(lines):
        if '[action]' in line and i + 1 < len(lines):
            action_trace.append(lines[i+1])
    
    return "\n".join(action_trace)


def extract_step_labels(output_str, correct_token, incorrect_token):
    step_labels = []
    # Find all boxed tokens (correct or incorrect)
    pattern = r'\\boxed\{(' + re.escape(correct_token) + '|' + re.escape(incorrect_token) + r')\}'
    matches = re.findall(pattern, output_str)
    
    for match in matches:
        if match == correct_token:
            step_labels.append(1)
        elif match == incorrect_token:
            step_labels.append(0)
            
    return step_labels