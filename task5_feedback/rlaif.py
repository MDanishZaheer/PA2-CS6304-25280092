# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Course-supplied fixed pairwise judge with safe lazy loading and caching."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.data import repo_path
from common.models import clear_gpu, resolve_dtype

PAIRWISE_RUBRIC = """You are comparing two candidate solutions to the same math problem.

Evaluate mathematical correctness, internal consistency, relevance, and whether the stated final answer follows from the reasoning.
Do not prefer a response merely because it is longer, more confident, or more polished.
If neither response is meaningfully better, return TIE.

Return exactly one token: A, B, or TIE.

Problem:
{problem}

Candidate A:
{a}

Candidate B:
{b}

Preference:"""


class PairwiseAIJudge:
    """Fixed course judge with deterministic A/B orientation balancing and persistent caching."""

    def __init__(self, cfg: dict, cache_path: str | Path):
        self.cfg = cfg
        self.cache_path = repo_path(cache_path)
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}
        self.tokenizer = None
        self.model = None
        self.cache_hits = 0
        self.model_calls = 0
        self.model_load_seconds = 0.0
        self.new_inference_seconds = 0.0

    def _load_model(self) -> None:
        """Load the 3B judge only after an uncached comparison is requested."""
        if self.model is not None:
            return
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["ai_judge_model"], padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available() and bool(
            self.cfg.get("quantize_frozen_models", True)
        ):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(
                    self.cfg.get("dtype", "float16")
                ),
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = (
                resolve_dtype(self.cfg.get("dtype", "float16"))
                if torch.cuda.is_available()
                else torch.float32
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["ai_judge_model"], **kwargs
        )
        if torch.cuda.is_available() and "device_map" not in kwargs:
            self.model = self.model.cuda()
        self.model.eval()
        self.model_load_seconds += time.perf_counter() - started

    def _orientation_key(self, problem, a, b):
        """Preserve the released deterministic A/B orientation rule."""
        payload = json.dumps(
            {"model": self.cfg["ai_judge_model"], "problem": problem, "a": a, "b": b},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _key(self, problem, a, b):
        """Invalidate cache entries if the fixed rubric or decoding changes."""
        payload = {
            "orientation_key": self._orientation_key(problem, a, b),
            "rubric_sha256": hashlib.sha256(PAIRWISE_RUBRIC.encode()).hexdigest(),
            "max_input_length": int(self.cfg["task5_judge_max_input_length"]),
            "max_new_tokens": int(self.cfg["task5_judge_max_new_tokens"]),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    @staticmethod
    def _parse_preference(text: str) -> tuple[str, bool]:
        match = re.search(r"\b(A|B|TIE)\b", str(text).strip().upper())
        return (match.group(1), False) if match else ("TIE", True)

    @torch.inference_mode()
    def compare_detail(self, problem: str, a: str, b: str) -> dict:
        """Return a traceable comparison while preserving starter semantics."""
        key = self._key(problem, a, b)
        if key in self.cache:
            self.cache_hits += 1
            saved = self.cache[key]
            if isinstance(saved, str):
                saved = {
                    "preference": saved,
                    "raw_output": "",
                    "swapped": None,
                    "parse_failure": False,
                    "input_tokens": None,
                    "inference_seconds": None,
                }
            return {**saved, "cache_key": key, "cache_hit": True}

        orientation_key = self._orientation_key(problem, a, b)
        swap = int(orientation_key[:8], 16) % 2 == 1
        aa, bb = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
        self._load_model()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        input_tokens = int(encoded["input_ids"].shape[1])
        maximum = int(self.cfg["task5_judge_max_input_length"])
        if input_tokens > maximum:
            raise ValueError(
                f"Task 5 judge input has {input_tokens} tokens; limit is {maximum}."
            )
        device = self.model.get_input_embeddings().weight.device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        started = time.perf_counter()
        out = self.model.generate(
            **encoded,
            max_new_tokens=int(self.cfg["task5_judge_max_new_tokens"]),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - started
        prompt_width = encoded["input_ids"].shape[1]
        decoded = self.tokenizer.decode(
            out[0, prompt_width:], skip_special_tokens=True
        ).strip()
        result, parse_failure = self._parse_preference(decoded)
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]

        detail = {
            "preference": result,
            "raw_output": decoded,
            "swapped": swap,
            "parse_failure": parse_failure,
            "input_tokens": input_tokens,
            "inference_seconds": elapsed,
        }
        self.cache[key] = detail
        self._save_cache()
        self.model_calls += 1
        self.new_inference_seconds += elapsed
        return {**detail, "cache_key": key, "cache_hit": False}

    def compare(self, problem: str, a: str, b: str):
        return self.compare_detail(problem, a, b)["preference"]

    def group_rewards(self, problem: str, responses: list[str]):
        k = len(responses)
        wins = np.zeros(k, dtype=float)
        ties = np.zeros(k, dtype=float)
        for i, j in itertools.combinations(range(k), 2):
            pref = self.compare(problem, responses[i], responses[j])
            if pref == "A":
                wins[i] += 1
            elif pref == "B":
                wins[j] += 1
            else:
                ties[i] += 1
                ties[j] += 1
        return ((wins + 0.5 * ties) / max(k - 1, 1)).tolist()

    def usage(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "model_calls": self.model_calls,
            "model_load_seconds": self.model_load_seconds,
            "new_inference_seconds": self.new_inference_seconds,
        }

    def close(self) -> None:
        if self.model is not None:
            model = self.model
            tokenizer = self.tokenizer
            self.model = None
            self.tokenizer = None
            clear_gpu(model, tokenizer)
