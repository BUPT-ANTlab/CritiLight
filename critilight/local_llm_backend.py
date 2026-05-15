import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalLLMBackend:
    def __init__(self, model_path=None):
        resolved_model_path = model_path or os.environ.get(
            "CRITILIGHT_LLM_MODEL_PATH",
            str(Path(__file__).resolve().parent / "model"),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            local_files_only=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            local_files_only=True,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_inputs(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.device)

    def infer(self, prompt):
        input_ids = self._build_inputs(prompt)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[:, input_ids.shape[-1]:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
