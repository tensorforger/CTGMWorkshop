from ctgmworkshop.architectures.flow_refractor.scaled_up_deeperflow import (
    FlowRefractorModel,
)

EXPERIMENT_NAME = "run_20_rectification"

import re
import os
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from diffusers.utils import load_image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2

from ctgmworkshop.flux_tools import *
from ctgmworkshop.image_tools import *


class FlowRefractorDataset(Dataset):
    def __init__(self):
        self.all_cond_latents_B = torch.load(
            "precomputes/flow_refractor/test-7/all_cond_latents_A.pt"
        )
        self.all_cond_latents_A = torch.load(
            "precomputes/flow_refractor/test-7/all_cond_latents_B.pt"
        )
        self.all_final_latents_B = torch.load(
            "precomputes/flow_refractor/test-7/all_final_latents_A_rect.pt"
        )
        self.all_final_latents_A = torch.load(
            "precomputes/flow_refractor/test-7/all_final_latents_B.pt"
        )
        self.all_init = torch.load("precomputes/flow_refractor/test-7/all_init_rect.pt")

    def __len__(self):
        return len(self.all_cond_latents_A)

    def __getitem__(self, idx):
        return (
            self.all_init[idx],
            self.all_cond_latents_A[idx],
            self.all_cond_latents_B[idx],
            self.all_final_latents_A[idx],
            self.all_final_latents_B[idx],
        )


def main():
    train_batch_size = 16

    device = "cuda"
    plt.style.use("dark_background")

    vae = AutoencoderKLFlux2.from_pretrained("FLUX.2-klein-4B/vae", device=device).to(
        device
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "FLUX.2-klein-4B/scheduler", device=device
    )

    vae_mean, vae_std = get_latents_mean_std(vae)

    dataset = FlowRefractorDataset()
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    model = FlowRefractorModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_train_timesteps = 1000

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params / 1000000, "M")

    state_dict = torch.load(
        "trained_models/flow_refractor/run_19_flow_refractor_l.pth",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    # if total_params > 100_000_000:
    #     raise ValueError(
    #         "The number of model's parameters is > 100M. Please decrease it and re-run."
    #     )

    def _get_next_sample_index(save_directory):
        """
        Finds the next available integer index for sample files
        of the form sample_<index>.png or sample_epochX_<index>.png.
        """
        pattern = re.compile(r"sample_(?:epoch\d+_)?(\d+)\.png")
        max_index = -1

        for fname in os.listdir(save_directory):
            match = pattern.match(fname)
            if match:
                max_index = max(max_index, int(match.group(1)))

        return max_index + 1

    def latents_to_np(latent):
        img = denorm_unpatchified_latents(latent, mean=vae_mean, std=vae_std)
        img = decode_latents(img, vae)
        img = denorm_image(img)
        return torch_to_np(img)

    @torch.no_grad()
    def test_model(
        save_directory,
        epoch_number=None,
        save=True,
    ):
        os.makedirs(save_directory, exist_ok=True)

        sample_index = _get_next_sample_index(save_directory)

        if epoch_number is not None:
            filename = f"sample_epoch{epoch_number}_{sample_index}.png"
        else:
            filename = f"sample_{sample_index}.png"

        save_path = os.path.join(save_directory, filename)

        init, cond_A, cond_B, final_A, final_B = next(iter(train_loader))

        cond_A = cond_A[:1].cuda().float()
        cond_B = cond_B[:1].cuda().float()
        final_A = final_A[:1].cuda().float()

        scheduler.set_timesteps(32, mu=1.0)
        latents = torch.normal(
            0, 1, (1, 32, 64, 120), dtype=torch.float32, device="cuda"
        )

        model.eval()

        for t in scheduler.timesteps:
            latent_model_input = latents
            t = t.to(device).view(1)

            predicted_noise = model(
                sample=latent_model_input,
                timestep=t,
                frame_A=cond_A,
                frame_B=cond_B,
                edited_frame_A=final_A,
            )

            latents = scheduler.step(predicted_noise, t, latents).prev_sample

        cond_image_A, cond_image_B, final_image_A, generated = map(
            latents_to_np, [cond_A, cond_B, final_A, latents]
        )

        fig, axes = plt.subplots(2, 2, figsize=(24, 18))

        images = [
            (cond_image_A, "Input A"),
            (cond_image_B, "Input B"),
            (final_image_A, "Edited A"),
            (generated, "Generated"),
        ]

        for ax, (img, title) in zip(axes.flat, images):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        if save:
            fig.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=150,
            )

        plt.close(fig)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"runs/flow_refractor/{EXPERIMENT_NAME}")

    epochs = 1000
    idx = 0

    loss_list = []

    for epoch in range(0, epochs):
        num_steps = 0
        loss_sum = 0
        model.train()
        for init, cond_A, cond_B, final_A, final_B in tqdm(train_loader):
            num_steps += 1

            batch_size = cond_A.shape[0]

            init = init.cuda().float()
            cond_A = cond_A.cuda().float()
            cond_B = cond_B.cuda().float()
            final_A = final_A.cuda().float()
            final_B = final_B.cuda().float()

            sigmas = torch.rand(batch_size, device=device)
            timesteps = sigmas * num_train_timesteps
            sigmas = sigmas[:, None, None, None]

            noise = init  # torch.normal(0, 1, cond_A.shape).cuda().float()

            noisy_latents = final_B * (1 - sigmas) + noise * sigmas

            predicted_velocity = model(
                sample=noisy_latents,
                timestep=timesteps,
                frame_A=cond_A,
                frame_B=cond_B,
                edited_frame_A=final_A,
            )

            target = noise - final_B

            loss = ((target - predicted_velocity) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())

            if num_steps == 100:
                model.eval()
                test_model(save_directory=f"samples/flow_refractor/{EXPERIMENT_NAME}")
                model.train()
                average_loss = loss_sum / num_steps
                loss_list.append(average_loss)
                writer.add_scalar("train_loss", average_loss, idx)
                num_steps = 0
                loss_sum = 0
                idx += 1

        torch.save(
            model.state_dict(), f"trained_models/flow_refractor/{EXPERIMENT_NAME}.pth"
        )

    print("Loss during training:")
    for i, l in enumerate(loss_list):
        print(f"idx = {i}, loss = {l:.5f}")

    print(f"Minimal loss: {min(loss_list):.5f}")


if __name__ == "__main__":
    main()
