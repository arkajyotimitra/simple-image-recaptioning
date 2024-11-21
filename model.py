import torch
from vllm import LLM, SamplingParams
from config import (
    LLAVA_PROMPT, QWEN_VL_PROMPT,
    LLAVA_CKPT_ID, QWEN_VL_CKPT_ID,
)

CKPT_ID_MAP = {
    "llava": LLAVA_CKPT_ID,
    "qwen-vl": QWEN_VL_CKPT_ID,
}
PROMPT_MAP = {
    "llava": LLAVA_PROMPT,
    "qwen-vl": QWEN_VL_PROMPT,
}

def load_vllm_engine(model_name="llava", max_tokens=120):
    devices = torch.cuda.device_count()
    vllm_engine = LLM(CKPT_ID_MAP[model_name], tensor_parallel_size=devices)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return vllm_engine, sampling_params


def infer(vllm_engine, inputs, model_name="llava"):
    sampling_params = inputs.pop("sampling_params")
    vllm_inputs = [{"prompt": PROMPT_MAP[model_name], "multi_modal_data": {"image": image}} for image in inputs["original_images"]]
    outputs = vllm_engine.generate(vllm_inputs, sampling_params)
    return outputs
