# Depth Anything: Analysis and Finetuning in Atypical Conditions

A **metric depth estimation** project built on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Metric Outdoor), finetuned on challenging scenarios and extended with **uncertainty estimation** via Deep Ensembles.

**Authors:** Davide Romano, Luca Saggiomo, Matteo Saccone

---

## The Problem

In safety-critical contexts (robotics, augmented reality, autonomous driving), depth estimates must be **precise and expressed in meters**. Depth Anything V2 Metric Outdoor performs well in typical conditions, but **fails in atypical ones**:

- adverse weather (rain, snow, fog)
- debris
- poor lighting
- aggressive camera motion

## Goals

1. **Finetune** the model to improve predictions in these conditions.
2. Estimate the **confidence** of its predictions, to know when the model can be trusted.

## Dataset

We use [**TartanAir**](https://theairlab.org/tartanair-dataset/), a photorealistic synthetic dataset with multimodal ground truth, designed to test Visual SLAM robustness in extreme edge cases.

| Split | Scenes |
| --- | --- |
| Train | Neighborhood, Season Forest Winter, Abandoned Factory Night, Western Desert |
| Validation | Gascola, Oldtown |
| Test | End of World |

**Known issues:** frames come from continuous video sequences, so they are highly correlated and lack diversity within a scene. To address this we used:

- fixed-stride temporal subsampling
- aggressive data augmentation

## Results

Evaluated on the test scene with three metrics: AbsRel, RMSE (lower is better) and δ₁ (higher is better).

| Metric | Base model | Finetuned model |
| --- | --- | --- |
| AbsRel | 0.95 | **0.15** |
| RMSE | 21.64 | **2.79** |
| δ₁ | 0.12 | **0.71** |

On **real-world** (non-synthetic) images, the finetuned model also shows encouraging qualitative results.

## Uncertainty Estimation: Deep Ensemble

To measure how much the predictions can be trusted, we trained an **ensemble of 5 models** with small variations, from which we derive a mean (the prediction) and a standard deviation (the **confidence map**).

Models were diversified through:

- different learning rate schedulers (fixed and cosine annealing)
- different warmup periods
- different gradient-loss weights
- different random seeds

The model shows higher uncertainty around object **edges**.

## Future Work

- Train on non-synthetic datasets such as SubT-MRS (born from the DARPA SubT Challenge)
- Edge AI optimization
- More reliable confidence maps, by moving to probabilistic training with Negative Log-Likelihood (NLL) losses

## References

- Yang et al., *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, CVPR 2024
- Yang et al., *Depth Anything V2*, NeurIPS 2024
- Wang et al., *TartanAir: A Dataset to Push the Limits of Visual SLAM*, IROS 2020
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NIPS 2017

---

*s2r, 2026*
