import math
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import tqdm as _tqdm
from diffusers.utils.torch_utils import randn_tensor

tqdm = partial(_tqdm.tqdm, dynamic_ncols=True)


VALID_SOLVERS: Tuple[str, ...] = ("flow", "dance", "ddim", "dpm1", "dpm2")


@dataclass
class DPMState:
    """Sliding window of recent model outputs for multistep DPM solvers."""

    order: int = 2
    model_outputs: List[Optional[torch.Tensor]] = field(default_factory=list)
    lower_order_nums: int = 0

    def __post_init__(self) -> None:
        if not self.model_outputs:
            self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor) -> None:
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self) -> None:
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1

def _broadcast_sigma(sigma_value: torch.Tensor, sample_shape) -> torch.Tensor:
    """Reshape a (B,) sigma tensor so it broadcasts against (B, *spatial)."""
    return sigma_value.view(-1, *([1] * (len(sample_shape) - 1)))


def _sigmas_for_timestep(
    scheduler, timestep, sample_shape
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Look up ``(σ_t, σ_{t+1}, σ_max)`` against a Flow-Match scheduler.

    Mirrors the (batched-timestep-tolerant) lookup used by
    ``sd3_sde_with_logprob.sde_step_with_logprob`` so this dispatcher is a
    drop-in replacement for that function.
    """
    if torch.is_tensor(timestep) and timestep.ndim > 0:
        step_index = [scheduler.index_for_timestep(t) for t in timestep]
    else:
        step_index = [scheduler.index_for_timestep(timestep)]
    prev_step_index = [s + 1 for s in step_index]

    sigma = _broadcast_sigma(scheduler.sigmas[step_index], sample_shape)
    sigma_prev = _broadcast_sigma(scheduler.sigmas[prev_step_index], sample_shape)
    sigma_max = scheduler.sigmas[1].item()
    return sigma, sigma_prev, sigma_max


def _sigmas_from_schedule(
    sigmas: torch.Tensor, index: int, sample_shape, device
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """``(σ_t, σ_{t+1}, σ_max)`` from an explicit sigma schedule (sampling-time)."""
    sigma = sigmas[index].to(device).view(*([1] * len(sample_shape)))
    sigma_prev = sigmas[index + 1].to(device).view(*([1] * len(sample_shape)))
    sigma_max = sigmas[1].item()
    # Broadcast across the batch dim; the math only needs the shape to
    # broadcast cleanly against `(B, C, H, W)` latents.
    sigma = sigma.expand(sample_shape[0], *([1] * (len(sample_shape) - 1)))
    sigma_prev = sigma_prev.expand(sample_shape[0], *([1] * (len(sample_shape) - 1)))
    return sigma, sigma_prev, sigma_max

def flow_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    eta: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma, sigma_prev, sigma_max = _sigmas_for_timestep(
        scheduler, timestep, sample.shape
    )
    dt = sigma_prev - sigma  # negative

    if eta > 0:
        sigma_max_t = torch.full_like(sigma, sigma_max)
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max_t, sigma)))
            * eta
        )
        prev_sample_mean = sample * (
            1 + std_dev_t**2 / (2 * sigma) * dt
        ) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        std_scale = std_dev_t * torch.sqrt(-1 * dt)
    else:
        # ODE Euler — deterministic.
        std_dev_t = torch.zeros_like(sigma)
        prev_sample_mean = sample + dt * model_output
        std_scale = std_dev_t

    if prev_sample is None:
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_scale * variance_noise
        else:
            prev_sample = prev_sample_mean

    if eta > 0:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * std_scale.clamp(min=1e-12) ** 2)
            - torch.log(std_scale.clamp(min=1e-12))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)

    return prev_sample, prev_sample_mean, std_dev_t, log_prob


def dance_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    eta: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma, sigma_prev, _ = _sigmas_for_timestep(scheduler, timestep, sample.shape)
    dt = sigma_prev - sigma  # negative
    delta_t = sigma - sigma_prev  # positive

    pred_original_sample = sample - sigma * model_output

    if eta > 0:
        std_dev_t = eta * torch.sqrt(torch.clamp(delta_t, min=0))
        score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (
            sigma**2
        ).clamp(min=1e-12)
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = sample + dt * model_output + log_term * dt
    else:
        std_dev_t = torch.zeros_like(sigma)
        prev_sample_mean = sample + dt * model_output

    if prev_sample is None:
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
        else:
            prev_sample = prev_sample_mean

    if eta > 0:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * std_dev_t.clamp(min=1e-12) ** 2)
            - torch.log(std_dev_t.clamp(min=1e-12))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)

    return prev_sample, prev_sample_mean, std_dev_t, log_prob


def ddim_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    eta: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma_s, sigma_t, _ = _sigmas_for_timestep(scheduler, timestep, sample.shape)
    s = sigma_s.to(torch.float64)
    t = sigma_t.to(torch.float64)

    # x0 prediction from a v-prediction-style model output (= sample - σ * pred)
    x0 = sample.to(torch.float64) - s * model_output.to(torch.float64)

    eps = (sample.to(torch.float64) - (1 - s) * x0) / s.clamp(min=1e-30)

    if eta > 0:
        sigma_dd = eta * t  # σ in DDIM paper (still in flow units here)
        dt_sqrt = torch.sqrt(
            torch.clamp(
                1.0
                - (t**2) * ((1 - s) ** 2) / ((s**2).clamp(min=1e-30) * ((1 - t) ** 2).clamp(min=1e-30)),
                min=0,
            )
        )
        rho_t = sigma_dd * dt_sqrt
        prev_mean = (1 - t) * x0 + torch.sqrt(torch.clamp(t**2 - rho_t**2, min=0)) * eps
    else:
        rho_t = torch.zeros_like(t)
        prev_mean = (1 - t) * x0 + t * eps

    prev_mean = prev_mean.to(torch.float32)
    rho_t = rho_t.to(torch.float32)

    if prev_sample is None:
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_mean + rho_t * variance_noise
        else:
            prev_sample = prev_mean

    if eta > 0:
        log_prob = (
            -((prev_sample.detach() - prev_mean) ** 2)
            / (2 * rho_t.clamp(min=1e-12) ** 2)
            - torch.log(rho_t.clamp(min=1e-12))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)

    return prev_sample, prev_mean, rho_t, log_prob


def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return 1 - sigma, sigma


def _dpm_first_order(
    scheduler, x0_pred: torch.Tensor, timestep, sample: torch.Tensor
) -> torch.Tensor:
    sigma_s, sigma_t, _ = _sigmas_for_timestep(scheduler, timestep, sample.shape)
    sigma_s = sigma_s.to(torch.float64)
    sigma_t = sigma_t.to(torch.float64)

    alpha_t, sig_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sig_s = _sigma_to_alpha_sigma_t(sigma_s)
    lam_t = torch.log(alpha_t.clamp(min=1e-30)) - torch.log(sig_t.clamp(min=1e-30))
    lam_s = torch.log(alpha_s.clamp(min=1e-30)) - torch.log(sig_s.clamp(min=1e-30))
    h = lam_t - lam_s

    x_t = (sig_t / sig_s.clamp(min=1e-30)) * sample.to(torch.float64) - alpha_t * (
        torch.exp(-h) - 1.0
    ) * x0_pred.to(torch.float64)
    return x_t.to(torch.float32)


def _dpm_second_order(
    scheduler,
    x0_pred_list: List[torch.Tensor],
    timestep_now,
    timestep_prev,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_s0, sigma_t, _ = _sigmas_for_timestep(scheduler, timestep_now, sample.shape)
    sigma_s1, _, _ = _sigmas_for_timestep(scheduler, timestep_prev, sample.shape)

    sigma_t = sigma_t.to(torch.float64)
    sigma_s0 = sigma_s0.to(torch.float64)
    sigma_s1 = sigma_s1.to(torch.float64)

    alpha_t, sig_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sig_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sig_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lam_t = torch.log(alpha_t.clamp(min=1e-30)) - torch.log(sig_t.clamp(min=1e-30))
    lam_s0 = torch.log(alpha_s0.clamp(min=1e-30)) - torch.log(sig_s0.clamp(min=1e-30))
    lam_s1 = torch.log(alpha_s1.clamp(min=1e-30)) - torch.log(sig_s1.clamp(min=1e-30))

    m0 = x0_pred_list[-1].to(torch.float64)
    m1 = x0_pred_list[-2].to(torch.float64)

    h = (lam_t - lam_s0).clamp(min=1e-30)
    h_0 = (lam_s0 - lam_s1).clamp(min=1e-30)
    r0 = h_0 / h
    D0 = m0
    D1 = (1.0 / r0.clamp(min=1e-30)) * (m0 - m1)

    x_t = (
        (sig_t / sig_s0.clamp(min=1e-30)) * sample.to(torch.float64)
        - alpha_t * (torch.exp(-h) - 1.0) * D0
        - 0.5 * alpha_t * (torch.exp(-h) - 1.0) * D1
    )
    return x_t.to(torch.float32)


def dpm_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    prev_sample: Optional[torch.Tensor] = None,
    order: int = 1,
    dpm_state: Optional[DPMState] = None,
    timestep_prev=None,
    is_last_step: bool = False,
):
    """Deterministic DPM-Solver++ step.

    ``order=1`` is a self-contained single-step update; ``order=2`` reuses
    the previous step's model output via ``dpm_state`` (along with
    ``timestep_prev`` for the second sigma lookup). Returns ``std_dev_t = 0``
    so the OPD KL falls back to the squared-L2 surrogate.

    Important — DPM-Solver++ operates on *data prediction* (``x0``), not
    velocity. The transformer outputs a v-prediction, so we convert here
    via ``x0_pred = sample - σ_s · v_pred`` before feeding it into the
    update formula or the multistep state buffer. This matches the
    DiffusionNFT reference (``solver.py::convert_model_output``) — passing
    the raw v-prediction would give a completely different, divergent
    trajectory and produce noise / blown-out images at sampling time.
    """
    model_output = model_output.float()
    sample = sample.float()

    sigma_s, _, _ = _sigmas_for_timestep(scheduler, timestep, sample.shape)
    x0_pred = sample - sigma_s.to(sample.dtype) * model_output

    if dpm_state is not None:
        dpm_state.update(x0_pred)

    use_first_order = (
        order == 1
        or dpm_state is None
        or dpm_state.lower_order_nums < 1
        or timestep_prev is None
        or is_last_step
    )
    if use_first_order:
        prev_mean = _dpm_first_order(scheduler, x0_pred, timestep, sample)
    else:
        prev_mean = _dpm_second_order(
            scheduler, dpm_state.model_outputs, timestep, timestep_prev, sample
        )

    if dpm_state is not None:
        dpm_state.update_lower_order()

    if prev_sample is None:
        prev_sample = prev_mean

    std_dev_t = torch.zeros(
        sample.shape[0],
        *([1] * (len(sample.shape) - 1)),
        device=sample.device,
        dtype=sample.dtype,
    )
    log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)

    return prev_sample, prev_mean, std_dev_t, log_prob


def solver_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    solver: str,
    noise_level: float,
    deterministic: bool = False,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    dpm_state: Optional[DPMState] = None,
    timestep_prev=None,
    is_last_step: bool = False,
):
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver!r}. Expected one of {VALID_SOLVERS}."
        )
    eta = 0.0 if deterministic else float(noise_level)

    if solver == "flow":
        return flow_step(
            scheduler,
            model_output,
            timestep,
            sample,
            eta=eta,
            prev_sample=prev_sample,
            generator=generator,
        )
    if solver == "dance":
        return dance_step(
            scheduler,
            model_output,
            timestep,
            sample,
            eta=eta,
            prev_sample=prev_sample,
            generator=generator,
        )
    if solver == "ddim":
        return ddim_step(
            scheduler,
            model_output,
            timestep,
            sample,
            eta=eta,
            prev_sample=prev_sample,
            generator=generator,
        )
    if solver == "dpm1":
        return dpm_step(
            scheduler,
            model_output,
            timestep,
            sample,
            prev_sample=prev_sample,
            order=1,
        )
    if solver == "dpm2":
        return dpm_step(
            scheduler,
            model_output,
            timestep,
            sample,
            prev_sample=prev_sample,
            order=2,
            dpm_state=dpm_state,
            timestep_prev=timestep_prev,
            is_last_step=is_last_step,
        )
    raise AssertionError("unreachable")

def run_sampling(
    v_pred_fn,
    z: torch.Tensor,
    scheduler,
    timesteps: torch.Tensor,
    solver: str = "flow",
    deterministic: bool = False,
    eta: float = 0.7,
    show_progress: bool = False,
    return_prev_sample_mean: bool = False,
    generator: Optional[torch.Generator] = None,
):
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver!r}. Expected one of {VALID_SOLVERS}."
        )
    dtype = z.dtype
    all_latents = [z]
    all_log_probs: List[torch.Tensor] = []
    all_prev_sample_means: List[torch.Tensor] = []

    dpm_state = DPMState(order=2) if solver == "dpm2" else None

    iterator = range(len(timesteps))
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Sampling Progress",
            disable=not (dist.is_initialized() and dist.get_rank() == 0),
        )

    last_step_idx = len(timesteps) - 1
    for i in iterator:
        t = timesteps[i]
        timestep_prev = timesteps[i - 1] if (i > 0 and solver == "dpm2") else None

        pred = v_pred_fn(z.to(dtype), t)

        z_next, prev_mean, _, log_prob = solver_step(
            scheduler=scheduler,
            model_output=pred.float(),
            timestep=t.unsqueeze(0) if t.ndim == 0 else t,
            sample=z.float(),
            solver=solver,
            noise_level=eta,
            deterministic=deterministic,
            prev_sample=None,
            generator=generator,
            dpm_state=dpm_state,
            timestep_prev=(timestep_prev.unsqueeze(0) if timestep_prev is not None and timestep_prev.ndim == 0 else timestep_prev),
            is_last_step=(i == last_step_idx),
        )
        z = z_next.to(dtype)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_prev_sample_means.append(prev_mean)

    if return_prev_sample_mean:
        return z, all_latents, all_log_probs, all_prev_sample_means
    return z, all_latents, all_log_probs
