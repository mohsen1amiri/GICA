"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import os
import threading
import requests
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer
from typing import List, Optional, Union

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Install it if you wish to use it as a model backend.")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def _generate_fastchat(
    prompt,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
    tokenizer,
    apply_chat_template=False,
    worker_addr="",
) -> ConcatedLMGenResult:
    if worker_addr == "":
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})  # get worker address by model name
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError("Language Model name {} does not exist.".format(model_name))

    prompt = prompt

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
    }
    try:
        response = requests.post(worker_addr + "/worker_generate", headers=headers, json=gen_params, stream=True)
        results = response.json()
    except Exception as e:
        print(f'Error in _generate_fastchat: {e}')

    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)]

    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )


@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 for vllm by default
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[Union[str, List[str]]] = None
    include_stop_str_in_output: bool = False


class LanguageModelCallingFunction:

    def __init__(self, llm_step_tag: str = None):
        self.llm_step_tag = llm_step_tag

    def __call__(self, messages: List, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError


class VLLMRemoteCaller(LanguageModelCallingFunction):

    def __init__(
        self,
        model_name,
        model_path,
        controller_addr="http://localhost:10014",
        llm_step_tag: str = None,
        is_critic: bool = False,
        apply_chat_template: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.controller_addr = controller_addr
        self.is_critic = is_critic
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.apply_chat_template = apply_chat_template
        super().__init__(llm_step_tag)

    def generate_response(self, prompt: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        return _generate_fastchat(
            prompt=prompt,
            model_name=self.model_name,
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            stop_token_ids=config.stop_token_ids,
            stop_str=config.stop_str,
            controller_addr=self.controller_addr,
            include_stop_str_in_output=config.include_stop_str_in_output,
            tokenizer=self.tokenizer,
            apply_chat_template=self.apply_chat_template,
        )


class LLMService:

    def __init__(
        self,
        model_name: str = "/cpfs02/user/liurunze/hf_models/models--meta-llama--Llama-3.1-8B-Instruct",
        model_type: str = "vllm",
        device: str = "cuda",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        use_tqdm: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.model_type = model_type.lower()
        self.dtype = dtype

        # self.load_lock = threading.Lock()
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_tqdm = use_tqdm

        self.pipe = None
        self.llm = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def start_service(self, max_model_len=16384):
        # with self.load_lock:
        if self.model_type == "hf":
            if self.pipe is None:
                print(f"Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype="auto",
                    device_map=self.device
                )
                print("Hugging Face model loaded successfully.")
        elif self.model_type == "vllm":
            if self.llm is None:
                print(f"Loading vLLM model '{self.model_name}' on device '{self.device}'...")
                self.llm = LLM(
                    self.model_name,
                    trust_remote_code=True,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=max_model_len,
                    dtype=self.dtype
                )
                print("vLLM model loaded successfully.")
        else:
            raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm' for vLLM.")

    def generate_response(
        self, prompt: str, n: int = 1, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1,
        max_tokens: int = 4096, stop: Optional[Union[str, List[str]]] = None, stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False
    ) -> List[str]:
        if self.model_type == "hf":
            return self._generate_response_hf(prompt, n, temperature, top_p, top_k, max_tokens, stop, stop_token_ids, include_stop_str_in_output)
        elif self.model_type == "vllm":
            return self._generate_response_vllm(prompt, n, temperature, top_p, top_k, max_tokens, stop, stop_token_ids, include_stop_str_in_output)
        else:
            raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm'.")

    def _generate_response_hf(
        self, prompt: str, n: int = 1, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1,
        max_tokens: int = 4096, stop: Optional[Union[str, List[str]]] = None, stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False
    ) -> List[str]:
        prompts = [prompt] * n
        responses = self.pipe(
            prompts,
            max_new_tokens=max_tokens,
            batch_size=n,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_full_text=False
        )
        response_message_batch = [result[0]["generated_text"] for result in responses]

        return response_message_batch, []

    def _generate_response_vllm(
        self, prompt: str, n: int = 1, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1,
        max_tokens: int = 4096, stop: Optional[Union[str, List[str]]] = None, stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False
    ) -> List[str]:
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        outputs = self.llm.generate(prompt, sampling_params=sampling_params, use_tqdm=self.use_tqdm)[0].outputs
        responses = [output.text for output in outputs]
        finish_reasons = [output.finish_reason for output in outputs]
        for idx, reason in enumerate(finish_reasons):
            if reason != "stop":
                pass
                # responses[idx] = '<INVALID>'

        return responses, finish_reasons


if __name__ == "__main__":
    # Initialize the service for vLLM
    llm_service = LLMService(model_type="vllm")
    llm_service.start_service()

    prompt = "What is game theory?"
    responses = llm_service.generate_response(prompt, n=3)

    print(responses)
