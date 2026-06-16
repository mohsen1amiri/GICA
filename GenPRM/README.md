<div align="center">

# GenPRM

Code derived from https://github.com/RyanLiu112/GenPRM
## 🎯 Overview

<img src="./static/images/fig_head.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="fig_head">




Create a new conda environment and install the dependencies:

```bash
conda create -n GenPRM python=3.10
conda activate GenPRM
pip install -r requirements.txt
```

### Examples & Demos

Try GenPRM in action with:

- **Interactive Jupyter Notebook**: [demo.ipynb](src/example/demo.ipynb) (quick start of GenPRM inference)
- **Process Supervision Cases**: [Case 1](src/example/case1.md) | [Case 2](src/example/case2.md)

For a quick start, you can use [gemprm_inference](src/prm_evaluation/gemprm_inference.py) module to implement model inference:

```python
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

genprm = GenPRM('GenPRM/GenPRM-7B')

messages = [ 
    {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."}, 
    {"role": "user", "content": "Question: Jo adds up all the positive integers from 1 to 100. Kate does a similar thing with the first 100 positive integers; however, she first rounds every integer to its nearest multiple of 10 (rounding 5s up) and then adds the 100 values. What is the positive difference between Jo's sum and Kate's sum?\n\nFirst, we need to calculate Jo's sum, which is the sum of all positive integers from 1 to 100. This can be directly computed using the formula for the sum of the first \\(n\\) positive integers, which is \\(\\frac{n(n+1)}{2}\\). For \\(n = 100\\), Jo's sum is \\(\\frac{100 \\cdot 101}{2} = 5050\\)."}, 
]
code_executor = CodeExecutor()
output, reward = genprm.inference(messages, cur_step=1, code_executor=code_executor)
print("Model output for the first solution step: " + output[0])
print(reward)
```
### Steps generate and Monte carlo score calculation

Generate policy steps

```bash
# example of math
bash reward_generation/steps_generate.sh \
    --LM models--Qwen--Qwen2.5-7B-Instruct \
    --round 0 \
    --bs 4 \
    --mt 6000 \
    --n_gpus 1 \
    --task math \
    --loop 1
```

Generate monte carlo scores

```bash
# example of math
bash reward_generation/mt_score_generate.sh \
    --LM models--Qwen--Qwen2.5-Math-7B-Instruct \
    --ORIGIN models--Qwen--Qwen2.5-7B-Instruct \
    --round 0 \
    --bs 4 \
    --mt 6000 \
    --n_gpus 1 \
    --task math \
    --loop 1
```

Generate reasoning data

```bash
# example of math
python rationale_generation/process.py \
    --model_path "Qwen/QwQ-32B" \
    --data_path _output/monte_carlo_processed/math_train_Qwen2.5-Math-7B-Instruct_monte_carlo \
    --save_path _output/reasoning_output/math_train_QwQ_reasoning \
    --num_gpu_per 1 \
    --majority_of_N 1
```

### Critique-refinement

Execute policy refinement based on GenPRM's split output

```bash
python prm_evaluation/policy_refine.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "_output/split_output/..."\
    --split_out "_output/split_refine/..."
```



> [!NOTE]
> Our mathematical expression evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). For a more powerful evaluator, please refer to this repository: [Math-Verify](https://github.com/huggingface/Math-Verify).



## 📝 Citation

If you find this work helpful, please kindly cite our paper:

```bibtex
@article{zhao2025genprm,
    title   = {GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning},
    author  = {Jian Zhao and Runze Liu and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    journal = {arXiv preprint arXiv:2504.00891},
    year    = {2025}
}
```

Our collection of PRMs in [Awesome-Process-Reward-Models](https://github.com/RyanLiu112/Awesome-Process-Reward-Models):

```bibtex
@misc{Awesome-Process-Reward-Models,
    title        = {Awesome Process Reward Models},
    author       = {Runze Liu and Jian Zhao and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    howpublished = {\url{https://github.com/RyanLiu112/Awesome-Process-Reward-Models}},
    note         = {GitHub repository},
    year         = {2025}
}
```

Our recent work on LLM test-time scaling with PRMs:

```bibtex
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```

## 💡 Acknowledgement

The model training is based on [axolotl](https://github.com/axolotl-ai-cloud/axolotl) and [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm). The mathematical evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).


