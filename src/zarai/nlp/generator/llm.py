"""LLM wrapper over Hugging Face transformers or vLLM."""

BACKENDS = {"hf", "vllm"}


class LLM:
    def __init__(self, model: str | None, backend: str = "hf", max_new_tokens=512, temperature=0.2):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(BACKENDS)}")
        self.model = model
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError(
            f"TODO: {self.backend} generation for model {self.model} (apply chat template, decode)"
        )


def load_llm(cfg) -> LLM:
    return LLM(cfg.model, cfg.backend, cfg.max_new_tokens, cfg.temperature)
