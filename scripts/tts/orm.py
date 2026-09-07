"""
orm.py
======

Outcome Reward Model (ORM) scorers for the TTS pipeline.

An ORM assigns ONE scalar score to a *complete* candidate solution, in contrast
to process-level verification, which scores every intermediate step.

NOTE: this is a stand-alone copy kept next to the drivers. The ORM drivers
import the packaged module instead (``from gica.tts.verifier.orm import
build_orm``), so edits made here have no effect on a run.
"""

from typing import List, Tuple

import torch


class ThinkPRMOutcomeORM:
    """Generative outcome-level scorer built on the repo's ThinkPRM wrapper."""

    def __init__(self, prm):
        """`prm` is an already-constructed gica.tts.verifier.ThinkPRM instance."""
        self.prm = prm

    def score_paths(self, question: str, paths: List[str], batch_size: int = 64) -> Tuple[List[float], int]:
        """Score every path with ONE outcome-level verifier call each.

        Returns (scores in [0,1], total generated verification tokens).
        """
        scores, gen_tokens = [], 0
        for lo in range(0, len(paths), batch_size):
            chunk = paths[lo:lo + batch_size]
            results = self.prm.predict_correctness_batch(
                questions=[question] * len(chunk),
                prefix_steps_batch=[[p] for p in chunk],  # whole path == single step
            )
            for res in results:
                s = float(res["prefix_score"])
                scores.append(s if s >= 0 else 0.0)  # -1 == unparsable -> worst score
                for out in res.get("outputs", []):
                    gen_tokens += len(self.prm.tokenizer.encode(out, add_special_tokens=False))
        return scores, gen_tokens

    def free(self):
        pass  # engine owned by the caller (may be reused for PRM stage)


class SeqClsORM:
    """Discriminative reward model: one forward pass per (question, solution)."""

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 4096,
                 dtype=torch.bfloat16, trust_remote_code: bool = False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code,
        ).to(device).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # SequenceClassification pools the last NON-PAD token: the model
        # config must know the pad id (Llama ships with None -> the
        # "batch sizes > 1" ValueError), and padding must be on the right
        # so the last real token is where the pooling logic expects it.
        self.tokenizer.padding_side = "right"
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def score_paths(self, question: str, paths: List[str], batch_size: int = 8) -> Tuple[List[float], int]:
        """Returns (raw reward logits, 0 generated tokens — discriminative)."""
        texts = []
        for p in paths:
            conv = [{"role": "user", "content": question},
                    {"role": "assistant", "content": p}]
            texts.append(self.tokenizer.apply_chat_template(conv, tokenize=False))
        scores = []
        for lo in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[lo:lo + batch_size], return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=self.max_length).to(self.device)
            logits = self.model(**enc).logits  # (B, 1) or (B, C)
            scores.extend(logits[:, 0].float().cpu().tolist())
        return scores, 0

    def free(self):
        del self.model
        torch.cuda.empty_cache()


class RLHFlowORM:
    """RLHFlow causal-LM ORM (e.g. RLHFlow/Llama3.1-8B-ORM-Deepseek-Data).

    Faithful implementation of RLHFlow's official `math-rm/orm_evaluate.py`:
      conversation = [{user: question + " " + solution}, {assistant: "+"}]
      logits at the "+"-prediction position (their published offset: 3rd
      token from the end for the Llama3.1-instruct template), softmax over
      the {'+','-'} token ids, score = P('+').

    Differences from their script (batch of 1) are efficiency-only and
    numerically exact: right padding with a per-row gather at
    (seq_len_i - offset), and the lm_head applied only to the gathered
    hidden states so the full (B, T, vocab) logits tensor is never
    materialized (Llama-3.1 vocab is 128k; full logits at batch 64 would
    be tens of GB).
    """

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 16384,
                 dtype=torch.bfloat16, offset: int = 3, strip_ki: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.max_length = max_length
        self.offset = offset          # "+" prediction position from the end
        self.strip_ki = strip_ki      # True only for the *-Mistral-Data models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        # exactly as in RLHFlow's orm_evaluate.py
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.plus_id = self.tokenizer.encode("+")[-1]
        self.minus_id = self.tokenizer.encode("-")[-1]
        self._offset_checked = False

    def _encode(self, question: str, path: str):
        ans = path.replace(" ки", "") if self.strip_ki else path
        conv = [{"content": question + " " + ans, "role": "user"},
                {"content": "+", "role": "assistant"}]
        ids = self.tokenizer.apply_chat_template(conv, return_tensors="pt")[0]
        if len(ids) > self.max_length:
            print(f"[RLHFlowORM] WARNING: sequence of {len(ids)} tokens exceeds "
                  f"max_length={self.max_length}; scoring it untruncated.")
        return ids

    @torch.no_grad()
    def score_paths(self, question: str, paths: List[str], batch_size: int = 8) -> Tuple[List[float], int]:
        batch_size = max(1, min(batch_size, 16))  # activation-memory guard for 8B
        encoded = [self._encode(question, p) for p in paths]
        pad_id = self.tokenizer.pad_token_id
        scores = []
        for lo in range(0, len(encoded), batch_size):
            chunk = encoded[lo:lo + batch_size]
            maxlen = max(len(x) for x in chunk)
            input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for i, ids in enumerate(chunk):
                input_ids[i, :len(ids)] = ids
                attn[i, :len(ids)] = 1
            input_ids, attn = input_ids.to(self.device), attn.to(self.device)

            lens = attn.sum(dim=1)
            pos = lens - self.offset  # per-row "+"-prediction position
            rows = torch.arange(len(chunk), device=self.device)

            # one-time sanity check: the token AFTER `pos` must be "+"
            if not self._offset_checked:
                tok_at_plus = input_ids[rows, pos + 1]
                if not bool((tok_at_plus == self.plus_id).all()):
                    print("[RLHFlowORM] WARNING: '+' not found at offset "
                          f"{self.offset} — your transformers version may "
                          "render the chat template differently; adjust the "
                          "`offset` argument (RLHFlow's published value is 3).")
                self._offset_checked = True

            try:  # memory-lean path: lm_head only on the gathered positions
                hidden = self.model.model(input_ids=input_ids,
                                          attention_mask=attn)[0]
                logits = self.model.lm_head(hidden[rows, pos])
            except AttributeError:  # non-standard architecture: full logits
                logits = self.model(input_ids=input_ids,
                                    attention_mask=attn).logits[rows, pos]
            cand = logits[:, [self.plus_id, self.minus_id]].float()
            scores.extend(cand.softmax(dim=-1)[:, 0].cpu().tolist())
        return scores, 0

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def build_orm(backend: str, orm_model: str = None, prm=None, **kw):
    if backend == "thinkprm":
        assert prm is not None, "thinkprm backend needs a ThinkPRM instance"
        return ThinkPRMOutcomeORM(prm)
    if backend == "seqcls":
        assert orm_model, "--orm_model is required for the seqcls backend"
        return SeqClsORM(orm_model, **kw)
    if backend == "rlhflow":
        assert orm_model, "--orm_model is required for the rlhflow backend"
        return RLHFlowORM(orm_model, **kw)
    raise ValueError(f"unknown ORM backend: {backend}")
