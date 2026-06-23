from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
import torch

device = "cuda"
dtype = torch.float16


def get_qwen3_prompt_embeds(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str | list[str],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_sequence_length: int = 512,
    hidden_states_layers: list[int] = (9, 18, 27),
):
    dtype = text_encoder.dtype if dtype is None else dtype
    device = text_encoder.device if device is None else device

    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    # Forward pass through the model
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    # Only use outputs from intermediate layers and stack them
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


def prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: torch.Tensor | None = None,
):
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)

    return torch.stack(out_ids)


@torch.no_grad()
def encode_prompt(
    prompt: str | list[str],
    text_encoder,
    tokenizer,
    device: torch.device | None = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: torch.Tensor | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int] = (9, 18, 27),
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embeds = get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_sequence_length=max_sequence_length,
        hidden_states_layers=text_encoder_out_layers,
    )

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    text_ids = prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(device)
    return prompt_embeds, text_ids


def prepare_latent_ids(
    latents: torch.Tensor,  # (B, C, H, W)
):
    r"""
    Generates 4D position coordinates (T, H, W, L) for latent tensors.

    Args:
        latents (torch.Tensor):
            Latent tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor:
            Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
            H=[0..H-1], W=[0..W-1], L=0
    """

    batch_size, _, height, width = latents.shape

    t = torch.arange(1)  # [0] - time dimension
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t, h, w, l)

    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return latent_ids


def prepare_image_ids(
    image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
    scale: int = 10,
):
    r"""
    Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

    This function creates a unique coordinate for every pixel/patch across all input latent with different
    dimensions.

    Args:
        image_latents (list[torch.Tensor]):
            A list of image latent feature tensors, typically of shape (C, H, W).
        scale (int, optional):
            A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
            latent is: 'scale + scale * i'. Defaults to 10.

    Returns:
        torch.Tensor:
            The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
            input latents.

    Coordinate Components (Dimension 4):
        - T (Time): The unique index indicating which latent image the coordinate belongs to.
        - H (Height): The row index within that latent image.
        - W (Width): The column index within that latent image.
        - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
    """

    if not isinstance(image_latents, list):
        raise ValueError(
            f"Expected `image_latents` to be a list, got {type(image_latents)}."
        )

    # create time offset for each reference image
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape

        x_ids = torch.cartesian_prod(
            t, torch.arange(height), torch.arange(width), torch.arange(1)
        )
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    image_latent_ids = image_latent_ids.unsqueeze(0)

    return image_latent_ids


def patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(
        batch_size, num_channels_latents * 4, height // 2, width // 2
    )
    return latents


def unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
    )
    return latents


def pack_latents(latents):
    """
    pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
    """

    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    return latents


def unpack_latents_with_ids(
    x: torch.Tensor,
    x_ids: torch.Tensor,
    height: int | None = None,
    width: int | None = None,
) -> list[torch.Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        # Use provided height/width to avoid DtoH sync from torch.max().item()
        h = height if height is not None else torch.max(h_ids) + 1
        w = width if width is not None else torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


def prepare_latents(
    batch_size,
    num_latents_channels,
    height,
    width,
    dtype,
    device,
    generator: torch.Generator,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.

    vae_scale_factor = 8
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)

    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    latent_ids = prepare_latent_ids(latents)
    latent_ids = latent_ids.to(device)

    latents = pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
    return latents, latent_ids


def get_latents_mean_std(vae):
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(device, dtype)

    return latents_bn_mean, latents_bn_std


@torch.no_grad()
def encode_vae_image(image: torch.Tensor, vae):
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")

    image = image.to(vae.device, vae.dtype)

    image_latents = vae.encode(image).latent_dist.sample()

    image_latents = patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        image_latents.device, image_latents.dtype
    )
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std

    return image_latents


def prepare_image_latents(
    images: list[torch.Tensor],
    vae,
    batch_size,
    device,
    dtype,
):
    image_latents = []
    for image in images:
        image = image.to(device=device, dtype=dtype)
        image_latent = encode_vae_image(image=image, vae=vae)
        image_latents.append(image_latent)  # (1, 128, 32, 32)

    image_latent_ids = prepare_image_ids(image_latents)

    # Pack each latent and concatenate
    packed_latents = []
    for latent in image_latents:
        # latent: (1, 128, 32, 32)
        packed = pack_latents(latent)  # (1, 1024, 128)
        packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
        packed_latents.append(packed)

    # Concatenate all reference tokens along sequence dimension
    image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
    image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

    image_latents = image_latents.repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.to(device)

    return image_latents, image_latent_ids


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: str | torch.device,
    sigmas: list[float],
    **kwargs,
):
    scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)

    return timesteps, num_inference_steps


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def norm_unpatchified_latents(latents, mean, std):
    latents = patchify_latents(latents)
    latents = latents - mean
    latents = latents / std
    latents = unpatchify_latents(latents)

    return latents


def denorm_unpatchified_latents(latents, mean, std):
    latents = patchify_latents(latents)
    latents = latents * std
    latents = latents + mean
    latents = unpatchify_latents(latents)

    return latents


@torch.no_grad()
def encode_latents(image, vae):
    """
    Encodes normalized torch image to denormalized latents
    """
    latents = vae.encode(image.to(vae.dtype)).latent_dist.sample()

    return latents


@torch.no_grad()
def decode_latents(latents, vae):
    """
    Decodes denormalized latents to normalized image
    """
    image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

    return image
