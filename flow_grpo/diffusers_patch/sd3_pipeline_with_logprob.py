# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py` for the legacy `solver='flow'` path.
# - It dispatches to the multi-solver `run_sampling` helper from `solver.py` for `solver in {'dance', 'ddim', 'dpm1', 'dpm2'}`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob
from .solver import VALID_SOLVERS, run_sampling

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    return_prev_sample_mean: bool = False,
    solver: str = "flow",
    deterministic: bool = False,
):
    """SD3 pipeline patched to (a) return per-step latents + log-probs and
    (b) dispatch the denoising loop through one of multiple samplers
    (``flow`` / ``dance`` / ``ddim`` / ``dpm1`` / ``dpm2``).

    The legacy ``solver='flow'`` path uses ``sde_step_with_logprob`` directly
    so existing OPD recipes are byte-for-byte unchanged. The other solvers
    delegate to ``flow_grpo.diffusers_patch.solver.run_sampling`` (adapted
    from DiffusionNFT). ``deterministic=True`` forces SDE-style solvers to
    use η = 0 (DPM solvers are deterministic by construction).
    """
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver!r}. Expected one of {VALID_SOLVERS}."
        )

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []
    all_prev_latents_mean = []

    # 7. Denoising loop
    if solver == "flow":
        # Legacy SDE path — unchanged from the pre-multi-solver code so
        # `solver='flow'` (= the default) reproduces existing OPD numbers
        # bit-for-bit.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                eta_active = 0.0 if deterministic else noise_level
                latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    noise_level=eta_active,
                )

                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_prev_latents_mean.append(prev_latents_mean)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
    else:
        # Multi-solver path. We hand a CFG-aware ``v_pred_fn`` to
        # ``run_sampling`` and let the solver dispatcher handle per-step
        # math + log-prob bookkeeping. The pipeline-side progress bar is
        # the helper's ``tqdm``; we keep it disabled by default so we
        # don't accidentally print two bars per call.
        prompt_embeds_local = prompt_embeds
        pooled_prompt_embeds_local = pooled_prompt_embeds
        do_cfg = self.do_classifier_free_guidance

        def v_pred_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            latent_model_input = torch.cat([z] * 2) if do_cfg else z
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds_local,
                pooled_projections=pooled_prompt_embeds_local,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            return noise_pred

        latents, traj_latents, traj_log_probs, traj_prev_means = run_sampling(
            v_pred_fn=v_pred_fn,
            z=latents,
            scheduler=self.scheduler,
            timesteps=timesteps,
            solver=solver,
            deterministic=deterministic,
            eta=noise_level,
            show_progress=False,
            return_prev_sample_mean=True,
            generator=None,
        )
        # `run_sampling` returns the *full* trajectory (including the
        # initial pre-step latent in `traj_latents[0]`), which already
        # matches what the legacy code path collected in `all_latents`.
        all_latents = traj_latents
        all_log_probs = traj_log_probs
        all_prev_latents_mean = traj_prev_means

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if return_prev_sample_mean:
        return image, all_latents, all_log_probs, all_prev_latents_mean
    return image, all_latents, all_log_probs
