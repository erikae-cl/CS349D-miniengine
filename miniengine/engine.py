"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper with orca like managing in terms of fixed allocation of memory."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_running = 16,
        max_seq_len = 2048,
    ):
        self.device = device
        self.dtype = dtype
        self.max_running = max_running
        self.max_seq_len = max_seq_len

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()

        # kv cache "pool"
        # dimensions is persistent (max_running, kv_heads, max_seq_len, head_dim) tensor
        # we need this per layer, per k and v. Each running request owns one slot row.
        # TODO adjust the hyper params of max_running, seq length based on mem capcities and workload
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_pool = [
            (
                torch.zeros(
                    max_running, self.num_kv_heads, max_seq_len, self.head_dim,
                    dtype=dtype, device=device,
                ),
                torch.zeros(
                    max_running, self.num_kv_heads, max_seq_len, self.head_dim,
                    dtype=dtype, device=device,
                ),
            )
            for i in range(self.num_layers)
        ]

        self.slot_to_request = {}
        self.next_free_slot = 0

        pool_bytes = (
            2  # K + V
            * self.num_layers
            * max_running
            * self.num_kv_heads
            * max_seq_len
            * self.head_dim
            * torch.tensor([], dtype=dtype).element_size()
        )
        logger.info(
            "kv pool stats:  max_running=%d, max_seq_len=%d, totalspace=%.2f GB",
            max_running, max_seq_len, pool_bytes / 1024**3,
        )

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # slot management, with compaction

    def alloc_slot(self, request):
        slot = self.next_free_slot
        self.slot_to_request[slot] = request
        self.next_free_slot += 1
        return slot

    def free_slot(self, slot_idx):
        last = self.next_free_slot - 1
        if slot_idx != last:
            moved_req = self.slot_to_request[last]
            L = moved_req.cache_len  # only copy valid portion
            # only move last to this and free last easy!
            for k_pool, v_pool in self.kv_pool:
                k_pool[slot_idx, :, :L, :].copy_(k_pool[last, :, :L, :])
                v_pool[slot_idx, :, :L, :].copy_(v_pool[last, :, :L, :])
            moved_req.slot_idx = slot_idx
            self.slot_to_request[slot_idx] = moved_req
        self.slot_to_request.pop(last, None)
        self.next_free_slot -= 1

    def _write_kv_to_pool(self, kv_caches, slot_idx, start_pos, seq_len):
        end_pos = start_pos + seq_len
        for layer_idx, (k, v) in enumerate(kv_caches):
            # k, v shape: (1, kv_heads, seq_len, head_dim)
            self.kv_pool[layer_idx][0][slot_idx, :, start_pos:end_pos, :] = k.squeeze(0)
            self.kv_pool[layer_idx][1][slot_idx, :, start_pos:end_pos, :] = v.squeeze(0)

    def _read_pool_slice(self, slot_idx, length):
        return [
            (
                self.kv_pool[layer_idx][0][slot_idx:slot_idx+1, :, :length, :],
                self.kv_pool[layer_idx][1][slot_idx:slot_idx+1, :, :length, :],
            )
            for layer_idx in range(self.num_layers)
        ]

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        process the full prompt in one forward pass, write the resulting kv
        into the request's pool slot, and sample the first output token.
        """
        # Validate length up front so we don't allocate a slot we can't fill.
        # Reserve 1 token of headroom for the first generated token.
        seq_len = len(request.input_ids)
        if seq_len + 1 > self.max_seq_len:
            raise ValueError(
                f"Prompt length {seq_len} (+1 for first decode) exceeds "
                f"max_seq_len={self.max_seq_len}. Increase --max-seq-len."
            )

        allocated_now = False
        if request.slot_idx is None:
            request.slot_idx = self.alloc_slot(request)
            request.cache_len = 0
            allocated_now = True

        try:
            input_ids = torch.tensor(
                [request.input_ids], dtype=torch.long, device=self.device
            )
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)

            self._write_kv_to_pool(kv_caches, request.slot_idx, start_pos=0, seq_len=seq_len)
            request.cache_len = seq_len

            return sample_token(
                logits[:, -1, :], request.sampling_params, request.output_ids
            )
        except Exception:
            # If anything failed after allocation, return the slot so the next
            # request doesn't see a phantom holder of this slot.
            if allocated_now:
                self.free_slot(request.slot_idx)
                request.slot_idx = None
                request.cache_len = 0
            raise

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        slot = request.slot_idx
        L = request.cache_len

        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        position_ids = torch.tensor([[L]], dtype=torch.long, device=self.device)

        # Vvew the request's existing cache out of the pool.
        kv_view = self._read_pool_slice(slot, length=L)

        logits, new_kv = self.model(input_ids, position_ids, kv_caches=kv_view)
        
        # append only 1 instead of whole!
        for layer_idx, (k_full, v_full) in enumerate(new_kv):
            self.kv_pool[layer_idx][0][slot, :, L, :] = k_full[0, :, -1, :]
            self.kv_pool[layer_idx][1][slot, :, L, :] = v_full[0, :, -1, :]
        request.cache_len = L + 1

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def batched_decode(self, requests):
        """
        runs one step of batched decode, assume stuff is prefilled,
        we do padding to the longest sequence in the batch and utilize
        masks to make sure we do not attend to pads.
        compaction used to reduce copies and do more reference passing
        """
        # sort by slot_idx so the i-th batch row corresponds to pool slot i.
        # like a scatter
        order = sorted(range(len(requests)), key=lambda i: requests[i].slot_idx)
        sorted_reqs = [requests[i] for i in order]
        batch_size = len(sorted_reqs)

        # after sort, slots should be 0,1,..n-1 thanks to compaction.
        # asserting just in case 
        assert all(r.slot_idx == j for j, r in enumerate(sorted_reqs)), (
            "asseert failed debug needed"
            f"Got: {[r.slot_idx for r in sorted_reqs]}"
        )

        last_tokens = [[r.output_ids[-1]] for r in sorted_reqs]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device)

        cache_lens = [r.cache_len for r in sorted_reqs]
        max_cache_len = max(cache_lens)
        cache_lens_t = torch.tensor(cache_lens, dtype=torch.long, device=self.device)
        position_ids = cache_lens_t.unsqueeze(1)  # (batch_size, 1)

        # view of tensors
        batched_kv_caches = []
        for layer_idx in range(self.num_layers):
            pool_k, pool_v = self.kv_pool[layer_idx]
            batched_kv_caches.append((
                pool_k[:batch_size, :, :max_cache_len, :],
                pool_v[:batch_size, :, :max_cache_len, :],
            ))

        # mask: (batch_size, 1, 1, max_cache_len + 1). +1 is the new token slot.
        mask = torch.zeros(
            batch_size, 1, 1, max_cache_len + 1,
            dtype=self.dtype, device=self.device,
        )
        # mask so attention score is 0 
        for i, L in enumerate(cache_lens):
            if L < max_cache_len:
                mask[i, :, :, L:max_cache_len] = float("-inf")

        # forward pass
        logits, new_kv_caches = self.model(
            input_ids, position_ids, batched_kv_caches, attention_mask=mask
        )

        # scatter the new kv token back to each slot's cache_len position.
        slots_t = torch.arange(batch_size, dtype=torch.long, device=self.device)
        for layer_idx in range(self.num_layers):
            new_k, new_v = new_kv_caches[layer_idx]
            new_k_token = new_k[:, :, -1, :]
            new_v_token = new_v[:, :, -1, :]
            pool_k, pool_v = self.kv_pool[layer_idx]
            pool_k[slots_t, :, cache_lens_t, :] = new_k_token
            pool_v[slots_t, :, cache_lens_t, :] = new_v_token

        for r in sorted_reqs:
            r.cache_len += 1

        # Sample in sorted order, then map back to caller's input order.
        sorted_tokens = [
            sample_token(logits[i:i+1, -1, :], r.sampling_params, r.output_ids)
            for i, r in enumerate(sorted_reqs)
        ]
        result = [None] * len(requests)
        for sorted_i, orig_i in enumerate(order):
            result[orig_i] = sorted_tokens[sorted_i]
        return result

    @torch.inference_mode()
    def mixed_step(self, ops):
        """
        mixed prefill+decode in a single forward pass.

        the forward pass uses batch=1 with all requests' new tokens
        concatenated along seq, plus a block-diagonal causal mask that
        keeps each request's queries restricted to its own keys.

        returns: per-op next token id, or None if the request is still
        prefilling (didn't finish prefill in this step).
        """
        if not ops:
            return []

        num_new_list = [n for r, n in ops]
        cache_lens = [r.cache_len for r, n in ops]
        slot_indices = [r.slot_idx for r, n in ops]
        total_new = sum(num_new_list)
        total_cached = sum(cache_lens)

        # concatenate input tokens. for prefilling requests pull next
        # chunk from input_ids. for decoding, use last output token.
        flat_tokens = []
        for req, n in ops:
            if req.cache_len < req.num_input_tokens:
                start = req.cache_len
                end = start + n
                flat_tokens.extend(req.input_ids[start:end])
            else:
                assert n == 1, "decoding requests must have num_new=1"
                flat_tokens.append(req.output_ids[-1])
        input_ids = torch.tensor([flat_tokens], dtype=torch.long, device=self.device)

        # each token's actual position in its OWN request.
        flat_positions = []
        for req, n in ops:
            for p in range(n):
                flat_positions.append(req.cache_len + p)
        position_ids = torch.tensor([flat_positions], dtype=torch.long, device=self.device)

        # gather cached kv from the pool, concat per layer along seq.
        cached_kv_caches = []
        for layer_idx in range(self.num_layers):
            pool_k, pool_v = self.kv_pool[layer_idx]
            ks = [pool_k[s:s+1, :, :L, :] for s, L in zip(slot_indices, cache_lens)]
            vs = [pool_v[s:s+1, :, :L, :] for s, L in zip(slot_indices, cache_lens)]
            cached_kv_caches.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))

        mask = self._build_mixed_mask(num_new_list, cache_lens)

        # forward
        logits, new_kv_caches = self.model(
            input_ids, position_ids, cached_kv_caches, attention_mask=mask
        )
        # logits: (1, total_new, vocab)
        # new_kv per layer: (1, kv_heads, total_cached + total_new, head_dim)

        # update kv pool
        for layer_idx in range(self.num_layers):
            new_k, new_v = new_kv_caches[layer_idx]
            new_k_only = new_k[:, :, total_cached:, :]   # (1, kv_heads, total_new, head_dim)
            new_v_only = new_v[:, :, total_cached:, :]
            pool_k, pool_v = self.kv_pool[layer_idx]
            offset = 0
            for slot, L, n in zip(slot_indices, cache_lens, num_new_list):
                pool_k[slot, :, L:L+n, :] = new_k_only[0, :, offset:offset+n, :]
                pool_v[slot, :, L:L+n, :] = new_v_only[0, :, offset:offset+n, :]
                offset += n

        # update cache_len + sample tokens for finished-prefill requests.
        results = []
        offset = 0
        for req, n in ops:
            req.cache_len += n
            if req.cache_len >= req.num_input_tokens:
                # prefill done (or already was decoding) so sample at the LAST
                # logit position belonging to this request.
                last_pos = offset + n - 1
                tok = sample_token(
                    logits[0:1, last_pos, :], req.sampling_params, req.output_ids
                )
                results.append(tok)
            else:
                #still prefilling, more chunks to come. no token to sample yet
                results.append(None)
            offset += n
        return results

    def _build_mixed_mask(self, num_new, cache_lens):
        """
        layout of the keys (along the K seq dim):
          [req0_cached, req1_cached, ..., reqN_cached,
           req0_new,    req1_new,    ..., reqN_new]
        layout of the queries (along the Q seq dim):
          [req0_new, req1_new, ..., reqN_new]

        each query position attends to: all positions in its own request's cached portion
        and its own request's new positions causally (k_pos <= q_pos)
        """
        total_new = sum(num_new)
        total_cached = sum(cache_lens)

        q_request = []
        q_position = []
        for r_idx, n in enumerate(num_new):
            for p in range(n):
                q_request.append(r_idx)
                q_position.append(p)
        q_request_t = torch.tensor(q_request, dtype=torch.long, device=self.device)
        q_position_t = torch.tensor(q_position, dtype=torch.long, device=self.device)

        k_cached_request = []
        for r_idx, L in enumerate(cache_lens):
            k_cached_request.extend([r_idx] * L)
        k_cached_request_t = torch.tensor(
            k_cached_request, dtype=torch.long, device=self.device,
        )

        # cached portion: true iff same request
        cached_attend = q_request_t[:, None] == k_cached_request_t[None, :]
        # New portion: same as Q (since new K and Q are the same positions).
        # Attend if same request AND k position <= q position (causal).
        same_req = q_request_t[:, None] == q_request_t[None, :]
        causal = q_position_t[:, None] >= q_position_t[None, :]
        new_attend = same_req & causal

        attend = torch.cat([cached_attend, new_attend], dim=1)
        mask = torch.where(attend, 0.0, float("-inf")).to(self.dtype)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_new, total_cached + total_new)

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
