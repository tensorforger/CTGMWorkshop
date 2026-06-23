# Flow Refractor architecture optimization

Your task is to optimize the architecture of the FlowRefractorModel.

# What the model does and what it is for

The model is an lightweight image editing flow matching model. Compared to large-scale image editing transformers, that take prompt with any instruction
and any reference images, the flow refractor takes only three images and no prompts. The images are:

1) Frame A: an original frame of the video. Just any frame, not edited, as is.
2) Frame B: an original frame of the video. It is sampled somewhere close to Frame A, usually 10-50 frames away.
3) Edited Frame A: The Frame A that was passed through image editing transformers with edit or restyle instruction.

Frame A and Edited Frame A are very close geometrically. Edges of objects usually align well.
Frame B is not aligned with them. But there are usually same objects, but shifted, like in common video.

The model should generate how Edited Frame B migth look if it was passed through same editing process.
The model is intended to be very lightweighted and run in real time. The large transformer can run on the background and generate a frame
per several seconds while our model warps it to current video stream.

All frames are passed to model as latents using Flux.2 VAE for encoding. Model should generate Edited Frame B as latents too.

# How the dataset looks like

This is important because you need to understand well what the model is optimizing for.

The dataset was generated this way:
1. Sample frames A and B from some video. In the current dataset videos are POV of long walking through some cities. They do not contain clips and
editing, just smooth motion. 
2. Process frame B through editing model. Yes, exactly B, not A. This way we can generate ground truth Edited Frame B very accurate without distortions.
3. Warp Edited Frame B towards geometry of frame A using optical flow and occlusion masking.
4. Use same editing model to fix warping artefacts: mostly remove occlusion holes. This would be Edited Frame A that would be passed to model.

Edited Frame A will be imperfect in some cases, but model will learn to fix this and undo the "warp -> mask -> fill" pipeline. 

All images are encoded to latents and stored as .pt files.

# What is your task

You only need to improve the architecture. The baseline is and SDXL-style unet with 91M parameters that simply concatenates all conditionings
along with noisy sample on input.

Your metric is the training loss after some fixed amount of training steps.

Do not simply tweak number of parametrs. Make radical changes that you think can solve this task better.

Some ideas to try (you are not forced to try exactly them):
1) Separate streams for (Frame A and Edited Frame A) and (Frame B and noisy sample), then cross attention between them.
2) FiLM style conditioning injection into denoising path
3) Grid sample inside the model's blocks. Model might need to do something like optical flow + warp but on internal feature maps.
4) RoPE on attention blocks
5) More down and up stages, thiner bottleneck

And of course you can try tweak number of channels and blocks, you can remove some modules and simplify.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 `Minimal loss` improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 `Minimal loss` improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

# Exact pipeline

Start with running the baseline training.

```bash
conda activate ctgmworkshop
python ./scripts/train_flow_refractor.py
```

It will do everything by itself: load dataset, train the model for number of steps required and print loss during training.
And most important: `Minimal loss` value. This is your optimization target. Try to make it as small as possible.


Then do in the loop:

1. Think on what do you want to change. Think strategically. Maybe some run will be worse then previous SOTA. But it might be insightful depending on the result.
2. Copy the file with architecture. Or make a new one. All files should be under `./src/ctgmworkshop/architectures/flow_refractor`. Make it descriptive name of whait is the architecture.
3. Edit this file and implement what you want to change.
4. Modify these two lines in `./scripts/train_flow_refractor.py`:
```
from ctgmworkshop.architectures.flow_refractor.baseline_unet import FlowRefractorModel
EXPERIMENT_NAME = "run_0_baseline"
```
5. Run `./scripts/train_flow_refractor.py`. It will show if some checks are not passed.
6. Add a block to `./report.md` (see more below)

For now you have **10** runs. You should not stop until you make 10 iterations. Do NOT pause to ask the human if you should continue.

You are completely autonomous. If something breaks you should try to fix it by your own. 

Stop and exit after 10 iterations are done. 

# Constraints

1. You can change architecture the way you want. Everything is fair. You can even edit previous runs in `./src/ctgmworkshop/architectures/flow_refractor` and see them.

This directory is completely yours.

2. Keep same model's API:

```python
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        frame_A: torch.Tensor,
        frame_B: torch.Tensor,
        edited_frame_A: torch.Tensor,
    ) -> torch.Tensor:
```

3. Don't exceed 100M parameter limit.
4. Don't change optimizer parameters, loss caclulation process, number of training steps, epochs and everything that can affect loss in `./scripts/train_flow_refractor.py`.
5. You can *slightly modify* other parts of `./scripts/train_flow_refractor.py`. For example, you can add additional logging if you need it.


# Reporting results

After each run add such block to `./report.md` (example for the baseline):

```
## Run 0: baseline

**Experiment name**: "run_0_baseline"

**Minimal loss**: *instert Minimal loss value*

**Architecture**: Unet with SDXL-style blocks. All conditionings are concatenated on input.

### Overview

- base_channels = 256
- 2 stages of down/up sampling
- block count per stage: 2 -> 4 -> 8 -> 4 -> 2
- chanllens per stage: 256 -> 512 -> 512 -> 512 -> 256
- attention: only in resolution/2 and resolution/4 blocks

### Motivation

SDXL is a good baseline. Let's start from it.

### Result analysis

This is the baseline, nothing to compare with.

```