# GptCompress

GptCompress explores how far open-weight language models can be compressed while preserving as much accuracy and useful behavior as possible. The project starts with Qwen3-4B and studies changes to its architecture, internal representations, and weights after training. The goal is a smaller model that can actually run, with measured trade-offs between size and quality.

## Research directions

- **Architecture.** Prune MLP channels, attention heads, or transformer blocks to see which parts of the model can be removed with the smallest quality loss.
- **Representations.** Factorize MLP and attention projection matrices into smaller low-rank factors. Explore shared bases across layers to reduce the amount of information stored repeatedly.
- **Weights.** Reduce weight precision and test whether compact corrections can recover enough accuracy to justify their storage cost.

Experiments are training-free. They can use forward-pass calibration and analytic reconstruction to guide compression. Individual methods and combinations are evaluated against the original model to see how much quality they retain at different sizes.

## Evaluation

The workspace records checkpoint size, task scores, runtime memory, and inference performance so changes can be compared under the same evaluation settings. The first baseline uses the original BF16 Qwen3-4B checkpoint on small subsets of GSM8K for math and IFEval for instruction following.

- [Evaluation setup and commands](Work/README.md)
- [Baseline report](Work/reports/EVAL-001.md)
- [Research brief and experiment requirements](Work/WORKER_REQUIREMENTS.md)
