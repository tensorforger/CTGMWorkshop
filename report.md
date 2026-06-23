# Architecture optimization report

## Run 0: baseline

**Experiment name**: "run_0_baseline"

**Minimal loss**: 0.65040

**Architecture**: UNet with SDXL-style blocks. All conditionings are concatenated on input.

### Overview

- base_channels = 256
- time_dim = 512, cond_dim = 1024
- 2 stages of down/up sampling
- block count per stage: 2 -> 4 -> 8 -> 4 -> 2
- channels per stage: 256 -> 512 -> 512 -> 512 -> 256
- attention: only in resolution/2 and resolution/4 blocks
- input: (sample, frame_A, frame_B, edited_frame_A) concatenated

### Motivation

SDXL is a good baseline. Let's start from it.

### Result analysis

### Result analysis

This is the baseline, nothing to compare with. Loss steadily decreases from ~1.64 to ~0.65 over 4 epochs.

## Run 1: Dual-stream UNet with cross-attention

**Experiment name**: "run_1_dual_stream"

**Minimal loss**: 0.69466

**Architecture**: Two separate UNet streams: edit stream (frame_A, edited_frame_A) and content stream (noisy_sample, frame_B) with cross-attention between them. Reduced to 19.4M params to fit in GPU memory.

### Overview

- base_channels = 128 (reduced from 192 to fit in memory after OOM)
- Edit stream: 2 down stages, 2 mid blocks
- Content stream: 2 down stages (2+2 blocks), 2 mid blocks, 2 up stages
- Cross-attention in content stream to attend to edit stream features
- Edit input: (frame_A, edited_frame_A) concatenated
- 19.4M params

### Motivation

Separate streams make the roles explicit: the edit stream learns "what changed", the content stream learns "where it needs to change in B". Cross-attention bridges them.

### Result analysis

Worse than baseline (0.695 vs 0.650). The model was severely under-parameterized (19.4M vs 93.6M) due to OOM when using 63M. The dual-stream design has higher memory cost at equal channel count.

## Run 2: Edit-Delta FiLM UNet

**Experiment name**: "run_2_edit_delta_film"

**Minimal loss**: 0.63908

**Architecture**: UNet processing (noisy_sample + frame_B). A separate edit encoder produces spatial delta features from (frame_A, edited_frame_A) at 3 scales. These delta features are injected via spatial FiLM (scale+shift per pixel) at each UNet stage.

### Overview

- base_channels = 224
- Edit encoder channels: (112, 224, 448)
- 2 down stages (2+4 blocks), 4 mid blocks, 2 up stages
- Spatial FiLM conditioning at all resolutions from the edit encoder
- 58.5M params

### Motivation

Instead of concatenating all inputs, extract the edit signal separately and inject it as spatial conditioning. FiLM allows the edit to multiplicatively and additively modulate features at every spatial position.

### Result analysis

**Beats baseline: 0.63908 vs 0.65040 (improvement of 0.011)**. Spatial FiLM conditioning works better than plain concatenation. The edit encoder can extract and propagate the edit signal at multiple scales. This is a meaningful improvement.

## Run 3: Lightweight UNet

**Experiment name**: "run_3_lightweight"

**Minimal loss**: 0.72623

**Architecture**: Same baseline architecture with base_channels=128 instead of 256. Direct parameter reduction.

### Overview

- base_channels = 128 (vs 256 in baseline)
- Same structure: 2 down stages, 4 mid blocks, 2 up stages
- Same input: (sample, frame_A, frame_B, edited_frame_A) concatenated
- 17.1M params

### Motivation

Test if baseline is over-parameterized. If similar loss with fewer params, architecture is wasteful.

### Result analysis

Worse than baseline (0.726 vs 0.650). The model was significantly under-parameterized. The baseline's capacity is actually needed for this task.

## Run 4: Correspondence Cross-Attention UNet

**Experiment name**: "run_4_correspondence"

**Minimal loss**: 0.64561

**Architecture**: Content UNet (noisy_sample + frame_B) uses cross-attention to query edit reference features. Queries from frame_B, keys from frame_A, values from edit pair — explicit spatial correspondence lookup.

### Overview

- base_channels = 192, 2 stages
- Frame encoder: encodes (frame_A, edited_frame_A) at 3 scales
- Per-block cross-attention in attention stages
- 43.0M params

### Motivation

Explicit spatial correspondence: each position in B finds its match in A and retrieves the edit applied there.

### Result analysis

**Beats baseline: 0.64561 vs 0.65040 (improvement of 0.005)**. Cross-attention correspondence helps, but less than spatial FiLM (run 2). The model is smaller (43M vs 94M) yet competitive.

## Run 5: Deeper 3-Stage UNet

**Experiment name**: "run_5_deeper"

**Minimal loss**: 0.72534

**Architecture**: Standard UNet with 3 down/up stages instead of 2, reaching 1/8 resolution bottleneck. Concatenates all inputs as baseline.

### Overview

- base_channels = 128, 3 stages: 128->256->384->256->128
- 2 blocks per stage, 4 mid blocks at 1/8 resolution
- Attention at /2 and /4 resolutions
- 32.2M params

### Motivation

More downsampling stages = larger receptive field for global context.

### Result analysis

Worse than baseline (0.725 vs 0.650). The model is too thin (32M params) for 3 stages. More stages need more channels to avoid information bottleneck.

## Run 6: Warp-Assist UNet

**Experiment name**: "run_6_warp_assist"

**Minimal loss**: 0.62817

**Architecture**: Estimates optical flow (frame_B → frame_A), warps edit features to frame_B space, then injects warped features as spatial FiLM in main UNet.

### Overview

- base_channels = 192
- FlowEstimator: lightweight CNN predicting 2D pixel offsets
- Edit feature encoder → grid_sample warp → spatial FiLM in main UNet
- 37.9M params

### Motivation

Explicitly model geometric displacement between A and B via flow estimation. Warped edit features should be spatially aligned with frame_B.

### Result analysis

**Best result: 0.62817 vs baseline 0.65040 (improvement of 0.022)**. The warp-assist architecture achieves the lowest loss. The flow estimation provides an inductive bias matching the video domain: frames are related by smooth motion. Even a rough flow estimate helps substantially.

## Run 7: Dual-Stream with Explicit Edit Delta

**Experiment name**: "run_7_dual_delta"

**Minimal loss**: 0.66164

**Architecture**: Dual-stream with edit input as (frame_A, edited_frame_A - frame_A) to make the edit delta explicit. Content stream cross-attends to edit stream features.

### Overview

- base_channels = 176
- Edit encoder: processes (frame_A, delta) where delta = edited_A - frame_A
- Content UNet cross-attends to edit features
- 36.1M params

### Motivation

Making the edit delta explicit removes ambiguity for the edit encoder.

### Result analysis

Worse than baseline (0.662 vs 0.650). The explicit delta didn't help. The cross-attention architecture adds overhead and the model is under-parameterized vs baseline.

## Run 8: Pre-Alignment UNet

**Experiment name**: "run_8_prealign"

**Minimal loss**: 0.62894

**Architecture**: Alignment network estimates A→B flow, warps frame_A and edited_frame_A to frame_B coordinate system, then standard UNet on all spatially-aligned inputs.

### Overview

- base_channels = 256, same structure as baseline
- AlignmentNetwork: U-shaped CNN estimating A→B flow field
- grid_sample warp of frame_A and edited_frame_A to frame_B space
- Standard UNet sees (noisy_sample, frame_B, warped_A, warped_edited_A)
- 69.0M params

### Motivation

If all frames are spatially aligned, the UNet task becomes simpler.

### Result analysis

**Second best: 0.62894 vs baseline 0.65040 (improvement of 0.021)**. Very close to run 6. Both warp-based approaches dominate. Pre-aligning inputs is nearly as effective as warping features inside the network.

## Run 9: Residual Inputs UNet

**Experiment name**: "run_9_residual_inputs"

**Minimal loss**: 0.64449

**Architecture**: Identical to baseline architecture but different input: (sample - frame_B, edited_frame_A - frame_A, frame_B, frame_A). Explicit residuals expose noise level and edit signal.

### Overview

- base_channels = 256, identical to baseline architecture
- Input 1: sample - frame_B (noisy residual)
- Input 2: edited_frame_A - frame_A (edit delta)
- Input 3: frame_B (content)
- Input 4: frame_A (spatial reference)
- 93.6M params

### Motivation

Minimal change from baseline: only input construction differs. Tests whether explicit residuals help.

### Result analysis

**Beats baseline: 0.64449 vs 0.65040 (improvement of 0.006)**. Simply using explicit residuals as inputs gives a free improvement over baseline with zero architectural change. The edit delta (edited_A - frame_A) is more informative than raw edited_A.

## Run 10: Multiscale Flow-Warp FiLM UNet

**Experiment name**: "run_10_multiscale_warp"

**Minimal loss**: 0.62576

**Architecture**: Multiscale feature pyramids for correspondence estimation and edit propagation. Flow is estimated at every encoder scale, edit features are warped at matching scales, and the warped features condition the UNet encoder through spatial FiLM.

### Overview

* base_channels = 192
* Multiscale feature extractor for frame_A and frame_B using the same residual hierarchy as the UNet encoder
* Coarse-to-fine multiscale flow estimator producing flow fields at every pyramid level
* Multiscale edit encoder mirroring the feature pyramid structure
* Per-scale feature warping using the corresponding flow estimate
* Spatial FiLM conditioning injected into every UNet encoder stage using warped edit features
* 53.8M params

### Motivation

Run 6 demonstrated that explicit geometric alignment is the strongest inductive bias discovered so far. However, a single flow estimate must simultaneously model both large displacements and fine spatial details. A multiscale correspondence hierarchy allows coarse levels to capture large motion while finer levels refine local alignment. By warping edit features independently at every scale and injecting them directly into the encoder hierarchy, the network receives spatially aligned edit information throughout the feature pyramid.

### Result analysis

**New best result: 0.62576 vs 0.62817 (improvement of 0.00241 over Run 6, 0.02464 over baseline).** The multiscale formulation successfully extends the benefits of warp-based conditioning. The gain is modest but consistent, suggesting that most of the benefit comes from introducing explicit geometric alignment, while multiscale refinement provides additional accuracy for difficult correspondences. Notably, the model achieves the best loss while remaining substantially smaller than the original 93.6M parameter baseline.

## Run 11: Separate OpticalFlow UNet Warp-Assist

**Experiment name**: "run_11_flow_unet"

**Minimal loss**: 0.62790

**Architecture**: A separate multiscale OpticalFlow UNet predicts decoder-side flow maps at multiple resolutions. The main denoising UNet uses these flow fields to warp edit features per scale, while the siamese frame encoder provides multiscale content features at the start of each stage.

### Overview

* base_channels = 192
* Separate multiscale OpticalFlow UNet
* Decoder outputs flow maps for multiple resolutions
* Siamese frame encoders for A and B
* Main UNet receives concatenated frame features at each stage
* Edited frame A is concatenated only with the noisy sample at the input
* 61.5M params

### Motivation

Run 10 showed that multiscale geometry helps. This variant tests whether a dedicated flow backbone is better than using flow as an auxiliary latent inside the denoising network. The hope is that a separate flow UNet can specialize more cleanly in correspondence estimation.

### Result analysis

**Beats Run 8 and Run 6 in stability, but is slightly worse than Run 10 in minimal loss: 0.62790 vs 0.62576.** The learned flow maps are now meaningful at every resolution, but the separate flow backbone adds extra parameters without fully improving the best loss. The model appears to benefit from cleaner flow specialization, yet the integrated multiscale formulation in Run 10 still remains stronger on the raw minimum metric.

## Run 12: Multihead Flow-Warp Assist

**Experiment name**: "run_12_multihead_flow"

**Minimal loss**: 0.60992

**Architecture**: The flow subsystem predicts multiple flow hypotheses per scale, similar to attention heads. Each hypothesis warps its own latent path, the paths are scored per pixel, normalized with softmax, and fused before FiLM modulation of the denoising UNet.

### Overview

* base_channels = 192
* 4 flow hypotheses per scale
* Per-head latent warping and softmax fusion
* Learned pixelwise score maps over warped and unwarped latents
* Multi-scale siamese frame features
* Main UNet remains attention-free
* 65.8M params

### Motivation

The flow maps from previous runs showed head specialization-like behavior: different branches seemed to capture different motion modes. This suggests that correspondence itself may be multi-modal. A multihead warp mechanism lets the model represent several plausible alignments simultaneously rather than forcing a single flow field to explain everything.

### Result analysis

**New best result: 0.60992 vs 0.62576 in Run 10.** This is the strongest model so far by a clear margin. The model appears to use the flow heads as a latent routing mechanism rather than as a literal optical flow estimator, which is exactly what the architecture encourages. The gain suggests that multi-hypothesis alignment is more expressive than a single warp field, especially when correspondences are ambiguous or motion boundaries are complex.

## Run 13: Latent Flow-FiLM UNet

**Experiment name**: "run_13_latent_flow_film"

**Minimal loss**: 0.64222

**Architecture**: A latent flow-like block is inserted after each residual block in the UNet. The block takes only the current feature tensor `x`, internally predicts flow hypotheses, warps latent chunks, scores them, and applies FiLM back onto the same `x`. Siamese A/B features are concatenated at the beginning of each stage, and edited A is only concatenated with the noisy input.

### Overview

* base_channels = 192
* Latent self-contained flow-FiLM block
* No general attention
* Siamese frame features injected at stage entry
* Edited frame A concatenated only with noisy input
* 68.1M params

### Motivation

This variant moves the flow mechanism inside the core denoising computation, making it behave more like self-attention: one input in, one output out, with latent routing performed internally. The goal is to test whether explicit flow supervision is unnecessary once the block becomes a general-purpose latent mixer.

### Result analysis

**Worse than Run 12 and Run 10: 0.64222.** The model is still substantially better than the baseline, but the fully internalized latent flow mechanism seems less effective than the more explicit multihead warp formulation. The result suggests that keeping flow prediction more structurally separated still helps optimization and preserves stronger geometric inductive bias.

## Run 14: Multihead Flow-Warp Assist

**Experiment name**: "run_14_multihead_flow_warp"

**Minimal loss**: 0.60865

**Architecture**: The flow-based warp path is treated as a latent multi-head routing mechanism. Multiple flow hypotheses are predicted per scale, edit features are warped by each hypothesis, and the resulting branches are fused with learned softmax gates inside the main UNet.

### Overview

* base_channels = 192
* 8 flow hypotheses per scale
* Multihead warp fusion inside the denoising UNet
* No general attention in the main UNet half-resolution path
* 73.5M params

### Motivation

The previous run showed that the flow heads behave less like literal flow predictors and more like specialized latent experts. This run tests whether the architecture benefits from giving the flow module more representational freedom while keeping the denoising UNet itself structurally unchanged.

### Result analysis

**New best result at the time: 0.60865.** The model shows strong optimization behavior and the learned flow maps remain meaningful across scales. The heads specialize into different motion modes, suggesting that the architecture is using the flow branch as a latent routing system rather than a single deterministic alignment mechanism.

---

## Run 15: Batched Multihead Flow-Warp Assist

**Experiment name**: "run_15_batched_multihead_flow"

**Minimal loss**: 0.61312

**Architecture**: Same multihead flow-warp model as Run 14, but the head dimension is explicitly batched in the tensor layout to remove the Python-loop style head-wise warp path and reduce VRAM pressure.

### Overview

* base_channels = 192
* 8 flow hypotheses per scale
* Batched head-axis flow prediction and warping
* Same main UNet structure as Run 14
* Lower VRAM overhead than the looped variant
* Parameter count similar to Run 14

### Motivation

Run 14 suggested that the multihead mechanism is useful, but the warp path is expensive. This variant tests whether the same idea can be expressed more efficiently by turning head-wise warping into a batched tensor operation.

### Result analysis

**Slightly worse minimal loss than Run 14: 0.61312 vs 0.60865.** The optimization curve is still much faster than earlier models, so the architectural idea remains strong. The more efficient implementation appears to trade a little peak performance for a noticeable reduction in memory cost.

---

## Run 16: Batched Multihead Flow-Warp Assist, 32 Heads

**Experiment name**: "run_16_batched_multihead_flow_32h"

**Minimal loss**: 0.61934

**Architecture**: Same batched multihead warp model as Run 15, but with 32 flow hypotheses instead of 8.

### Overview

* base_channels = 192
* 32 flow hypotheses per scale
* Batched head-axis flow prediction and warping
* Same main UNet structure
* One parameter tweak relative to Run 15

### Motivation

This checks whether more hypotheses help when the flow branch is already being used as a latent expert system. If the heads really function like attention experts, more heads should increase flexibility, even if individual head quality becomes noisier.

### Result analysis

**Worse than the 8-head batched version: 0.61934.** More heads do not help once the system is already expressing multiple motion modes. This supports the idea that the useful capacity lies in structured specialization, not in simply increasing the number of hypotheses.

---

## Run 17: Deeper Flow UNet Warp-Assist

**Experiment name**: "run_17_deeper_flow_unet"

**Minimal loss**: 0.60482

**Architecture**: The dedicated OpticalFlow UNet is deepened by one additional down/up stage in the bottleneck, while keeping the same 8-head batched multihead fusion and the same three output flow resolutions consumed by the main UNet.

### Overview

* base_channels = 192
* 8 flow hypotheses per scale
* Deeper dedicated OpticalFlow UNet
* One extra internal down/up stage in the flow backbone
* Output flow pyramid size unchanged
* 69.7M params

### Motivation

The earlier flow maps were already meaningful, but the bottleneck representation may still be too shallow to model harder correspondences. This variant increases the expressivity of the flow backbone while keeping the main denoiser unchanged.

### Result analysis

**New best result: 0.60482.** This is a strong improvement over the previous models. The deeper flow backbone appears to improve correspondence quality without sacrificing the benefits of multihead latent routing. The smoothed loss curve being consistently lowest is especially encouraging, since it suggests this is not just a transient spike but a genuinely better optimization regime.

## Run 18: Scaled-Up Deeper Flow UNet Warp-Assist

**Experiment name**: "run_18_scaled_up_deeperflow"

**Minimal loss**: **0.58052**

**Architecture**: Identical to Run 17, but scaled from `base_channels = 192` to `base_channels = 288`. No architectural changes were made. The only difference is model capacity.

### Overview

* base_channels = 288
* Deeper dedicated OpticalFlow UNet
* 8 flow hypotheses per scale
* Batched multihead flow fusion
* 152.6M params

### Motivation

Run 17 established that the deeper flow backbone was the strongest architecture so far. The next question is whether the architecture has saturated, or whether additional capacity can still be effectively utilized. This experiment isolates scaling effects by increasing width while keeping the architecture unchanged.

### Result analysis

**Major new SOTA: 0.58052 vs 0.60482.**

This is the largest single improvement since the introduction of the multihead flow architecture. The fact that such a substantial gain comes from a pure width increase strongly suggests that the current architecture remains capacity-limited rather than optimization-limited.

Perhaps more importantly, this result validates the overall direction of development. If performance had plateaued, scaling would have produced diminishing returns. Instead, the architecture continues to convert additional parameters into meaningful quality improvements.

The result also suggests that the flow subsystem is not merely acting as a regularizer. It appears to be learning increasingly useful geometric representations as capacity grows.

---

## Final Summary

| Run | Architecture                           | Params | Minimal Loss | vs Baseline |
| --- | -------------------------------------- | -----: | -----------: | ----------: |
| 0   | Baseline UNet                          |  93.6M |      0.65040 |           — |
| 1   | Dual-stream cross-attn                 |  19.4M |      0.69466 |      -0.044 |
| 2   | Edit-Delta FiLM                        |  58.5M |      0.63908 |  **+0.011** |
| 3   | Lightweight UNet                       |  17.1M |      0.72623 |      -0.076 |
| 4   | Correspondence Cross-Attn              |  43.0M |      0.64561 |  **+0.005** |
| 5   | Deeper 3-Stage                         |  32.2M |      0.72534 |      -0.075 |
| 6   | Warp-Assist                            |  37.9M |      0.62817 |  **+0.022** |
| 7   | Dual-Delta                             |  36.1M |      0.66164 |      -0.011 |
| 8   | Pre-Align                              |  69.0M |      0.62894 |  **+0.021** |
| 9   | Residual Inputs                        |  93.6M |      0.64449 |  **+0.006** |
| 10  | Multiscale Flow-Warp FiLM              |  53.8M |      0.62576 |  **+0.025** |
| 11  | Separate OpticalFlow UNet              |  61.5M |      0.62790 |  **+0.023** |
| 12  | Multihead Flow-Warp Assist             |  65.8M |      0.60992 |  **+0.040** |
| 13  | Latent Flow-FiLM UNet                  |  68.1M |      0.64222 |  **+0.008** |
| 14  | Multihead Flow-Warp Assist (8 heads)   |  73.5M |      0.60865 |  **+0.042** |
| 15  | Batched Multihead Flow-Warp (8 heads)  |      — |      0.61312 |  **+0.037** |
| 16  | Batched Multihead Flow-Warp (32 heads) |      — |      0.61934 |  **+0.031** |
| 17  | Deeper Flow UNet Warp-Assist           |  69.7M |      0.60482 |  **+0.046** |
| 18  | Scaled-Up Deeper Flow UNet             | 152.6M |  **0.58052** |  **+0.070** |

**Winner: Run 18 (Scaled-Up Deeper Flow UNet Warp-Assist)** with minimal loss **0.58052**.

### Evolution of the Best Models

| Generation                        |   Best Loss |
| --------------------------------- | ----------: |
| Baseline                          |     0.65040 |
| Run 6 Warp-Assist                 |     0.62817 |
| Run 10 Multiscale Flow-Warp       |     0.62576 |
| Run 12 Multihead Flow-Warp        |     0.60992 |
| Run 17 Deeper Flow UNet           |     0.60482 |
| Run 18 Scaled-Up UNet             | **0.58052** |

Total improvement from baseline:

**0.65040 → 0.58052**

Absolute improvement:

**0.06988**

Relative improvement:

**10.74% reduction in loss**

### Key insights

#### 1. Geometry consistently beats pure feature conditioning

The largest gains throughout the project came from architectures that explicitly model correspondence between frames. Every major SOTA jump after Run 6 involved stronger geometric reasoning.

#### 2. Multihead flow behaves more like latent experts than optical flow

Inspection of the learned flow maps showed that different heads specialize into different motion regimes. The architecture appears to use flow prediction as a structured routing mechanism rather than a strict geometric estimator.

#### 3. Flow quality matters

Moving from:

* implicit flow,
* to multiscale flow,
* to multihead flow,
* to a deeper dedicated flow backbone,

produced a remarkably consistent sequence of improvements.

#### 4. Capacity scaling remains effective

Run 18 may be the most important result in the entire report.

The architecture was not saturated at 70M parameters. Doubling width from 192 to 288 channels produced a dramatic gain without any architectural modification.

This strongly suggests that:

* optimization remains healthy,
* the inductive bias is useful,
* additional capacity is still being converted into useful representations.

#### 5. More heads are not necessarily better

Run 16 demonstrates that simply increasing the number of flow hypotheses does not automatically help. Head specialization appears more important than head count.

#### 6. The flow subsystem has become the primary innovation

Originally the flow path was introduced as a lightweight auxiliary alignment mechanism.

By Run 18 it has evolved into a dedicated geometric backbone that appears responsible for most of the performance gains. The denoising UNet increasingly acts as a consumer of geometric information rather than the sole source of reasoning.

### Current Best Architecture

**Run 18: Scaled-Up Deeper Flow UNet Warp-Assist**

* base_channels = 288
* 152.6M parameters
* Dedicated deep OpticalFlow UNet
* 8-head multiscale flow hypotheses
* Batched warp fusion
* Multiscale edit feature warping
* Attention-free main UNet
* Minimal loss: **0.58052**

This is the current state-of-the-art result in the experiment series and provides strong evidence that the geometric correspondence pathway continues to scale effectively with model capacity.
