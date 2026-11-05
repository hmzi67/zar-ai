"""Load a tiny LLaMA-architecture model in 4-bit on the GPU with bitsandbytes and run generate().

The model is randomly initialised and built locally, so nothing is downloaded. It shows the GPU
stack works, not how much VRAM a real model needs.
"""

import tempfile

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

VOCAB_SIZE = 1000


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: check nvidia-smi")

    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    with tempfile.TemporaryDirectory() as model_dir:
        LlamaForCausalLM(config).save_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )

    layers = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    weight = model.model.layers[0].self_attn.q_proj.weight
    if layers == 0 or not isinstance(weight, bnb.nn.Params4bit) or weight.device.type != "cuda":
        raise SystemExit(
            f"4-bit load failed: {layers} Linear4bit layers, weight on {weight.device}"
        )

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8)).cuda()
    output = model.generate(input_ids, max_new_tokens=4, do_sample=False)

    print(f"bitsandbytes {bnb.__version__}, transformers {transformers.__version__}")
    print(f"{layers} Linear4bit layers, {weight.quant_type} weights on {weight.device}")
    print(f"generated shape: {tuple(output.shape)}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
