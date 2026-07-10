# sm103 common Pitfalls (nvidia)

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| [flash_attention_prefill-pitfalls.md](flash_attention_prefill-pitfalls.md) | `flash_attention_prefill` — vLLM (pai b46dc08) ModelOpt NVFP4 path + FA4 attention, NVFP4 MoE weights + bf16 attention, qwen3.7-max / qwen3.5-plus (NVFP4 MoE, hd256), TP4/TP8 | B300 / Blackwell / sm103 | 1 |
| [paged_attention_decode-pitfalls.md](paged_attention_decode-pitfalls.md) | `paged_attention_decode` — flashinfer trtllm-gen (prebuilt cubin) baseline, bf16, 31 production shapes; failures concentrated on block256 / klen1024 大 shape | B300 / Blackwell / sm103 | 3 |
