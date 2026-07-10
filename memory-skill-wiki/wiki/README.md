# Kernel-Experience Wiki (gpu-wiki structured)

Distilled GPU-kernel-optimization experience, laid out to match `atrex-kernel-agent/gpu-wiki` (vendor → DSL → architecture). Each topic has up to three docs: **kernel-opt** (proven techniques), **pitfalls** (Trap→Result→Why→Lesson), and **ref-docs** (full version journey).

_47 experience record(s) across 6 topic group(s)._

| Vendor | DSL | Arch | Topic | Records | ✅ | ❌ | kernel-opt | pitfalls | journey |
|--------|-----|------|-------|--------:|---:|---:|-----------|----------|---------|
| nvidia | common | sm103 | flash_attention_prefill | 1 | 0 | 1 | — | [✓](docs/pitfalls/nvidia/common/sm103/flash_attention_prefill-pitfalls.md) | [✓](docs/ref-docs/nvidia/common/sm103/flash_attention_prefill-optimization.md) |
| nvidia | common | sm103 | paged_attention_decode | 3 | 0 | 3 | — | [✓](docs/pitfalls/nvidia/common/sm103/paged_attention_decode-pitfalls.md) | [✓](docs/ref-docs/nvidia/common/sm103/paged_attention_decode-optimization.md) |
| nvidia | common | sm90 | flash_attention | 1 | 0 | 0 | — | — | [✓](docs/ref-docs/nvidia/common/sm90/flash_attention-optimization.md) |
| nvidia | cutedsl | sm100 | flash_attention | 3 | 0 | 1 | — | [✓](docs/pitfalls/nvidia/cutedsl/sm100/flash_attention-pitfalls.md) | [✓](docs/ref-docs/nvidia/cutedsl/sm100/flash_attention-optimization.md) |
| nvidia | cutedsl | sm103 | flash_attention_prefill | 12 | 6 | 4 | [✓](docs/kernel-opt/nvidia/cutedsl/sm103/flash_attention_prefill.md) | [✓](docs/pitfalls/nvidia/cutedsl/sm103/flash_attention_prefill-pitfalls.md) | [✓](docs/ref-docs/nvidia/cutedsl/sm103/flash_attention_prefill-optimization.md) |
| nvidia | cutedsl | sm103 | paged_attention_decode | 27 | 12 | 8 | [✓](docs/kernel-opt/nvidia/cutedsl/sm103/paged_attention_decode.md) | [✓](docs/pitfalls/nvidia/cutedsl/sm103/paged_attention_decode-pitfalls.md) | [✓](docs/ref-docs/nvidia/cutedsl/sm103/paged_attention_decode-optimization.md) |
