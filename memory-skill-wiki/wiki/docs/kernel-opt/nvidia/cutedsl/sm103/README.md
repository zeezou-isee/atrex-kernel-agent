# sm103 cutedsl Optimization Highlights (nvidia)

| File | Kernel | Hardware | Techniques |
|------|--------|----------|-----------|
| [flash_attention_prefill.md](flash_attention_prefill.md) | `flash_attention_prefill` — CuTeDSL (cutlass 4.5.2), FA4 flash_attn.cute general FlashAttentionForwardSm100, bf16 in/out, fp32 accum, hd256, GQA (qwen3.7-max/3.5-plus), paged KV + varlen + seqused_k | B300 / Blackwell / sm103 / sm100a | 6 |
| [paged_attention_decode.md](paged_attention_decode.md) | `paged_attention_decode` — CuTeDSL (cutlass 4.5.2), bf16 in/out, fp32 partial/accum, hd256 (D=256), GQA, split-KV num_splits, combine 归约 | B300 / Blackwell / sm103 | 12 |
