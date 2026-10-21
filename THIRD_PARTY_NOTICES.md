# Attribution

Original MicroLlama V2 code: Zixiao Ken Wang, Apache 2.0. Retained third-party files keep their own license and provenance notices.

- Qwen3 architecture/components: [Qwen3](https://github.com/QwenLM/Qwen3), [technical report](https://arxiv.org/abs/2505.09388); implementation provided by Transformers.
- Engram inspiration: [official repository](https://github.com/deepseek-ai/Engram), [paper](https://arxiv.org/abs/2601.07372). This repository's lightweight prototype is not the full official architecture.
- Transformers: [Hugging Face](https://github.com/huggingface/transformers), Apache 2.0.
- Muon experimental optimizer: Keller Jordan; its MIT license and pinned provenance remain in `microllama/training/vendor/muon/`.
- XMemTransfer experimental work: retained license in `microllama/training/vendor/xmemtransfer/`.
- FineWeb-Edu, DCLM-Edu, Cosmopedia v2 and FineMath: source terms/provenance remain in the source inventories and public token archive. The code license does not replace upstream data terms.
- V1 lineage: [TinyLlama](https://arxiv.org/abs/2401.02385), LitGPT and FlashAttention. V1 citation: Ken Wang, *MicroLlama-300M*, 2024.
- Benchmark references: SmolLM2 and LiquidAI. Their prescribed citations are included in `CITATIONS.bib`.

- MLX/Metal experimental kernels and shared headers: Apple Inc., MIT; the retained license is `microllama/training/metal/MLX-LICENSE.txt`. Native stock MLX operations are the default; custom kernels remain optional research paths.
