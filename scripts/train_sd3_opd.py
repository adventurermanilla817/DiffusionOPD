from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.solver import VALID_SOLVERS, solver_step
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import wandb
from functools import partial, wraps
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def make_no_grad(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper

def _build_teachers_config(config):
    global_proxy = dict(getattr(config, "proxy_reward_fn", {}) or {})
    global_prompt_fn = getattr(config, "prompt_fn", "general_ocr")
    global_teacher_gs = float(
        getattr(
            config.sample,
            "teacher_guidance_scale",
            getattr(config.sample, "guidance_scale", 1.0),
        )
    )

    def _resolve_gs(value, default):
        return float(default if value is None else value)

    teachers = getattr(config.train, "teachers", None)
    if not teachers:
        raise ValueError("OPD training requires `config.train.teachers`.")

    out = []
    for t in teachers:
        entry = dict(t)
        for required in ("name", "dataset", "lora_path"):
            if required not in entry:
                raise ValueError(
                    f"Teacher entry {entry!r} is missing required key '{required}'."
                )
        entry["proxy_reward"] = dict(entry.get("proxy_reward") or global_proxy)
        entry["prompt_fn"] = entry.get("prompt_fn") or global_prompt_fn
        entry["guidance_scale"] = _resolve_gs(
            entry.get("guidance_scale"), global_teacher_gs
        )
        out.append(entry)

    names = [t["name"] for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Teacher slot names must be unique, got {names}.")
    return out

from accelerate.state import AcceleratorState

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:

            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:

        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def _resolve_solver(config):

    solver = str(getattr(config.sample, "solver", "flow"))
    deterministic = bool(getattr(config.sample, "deterministic", False))
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver!r}. Expected one of {VALID_SOLVERS}."
        )
    return solver, deterministic

def compute_student_step(transformer, pipeline, sample, j, embeds, pooled_embeds, config):

    solver, deterministic = _resolve_solver(config)

    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    if solver == "flow" and not deterministic:

        _, _, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            sample["timesteps"][:, j],
            sample["latents"][:, j].float(),
            prev_sample=sample["next_latents"][:, j].float(),
            noise_level=config.sample.noise_level,
        )
    else:
        _, prev_sample_mean, std_dev_t, _ = solver_step(
            scheduler=pipeline.scheduler,
            model_output=noise_pred.float(),
            timestep=sample["timesteps"][:, j],
            sample=sample["latents"][:, j].float(),
            solver=solver,
            noise_level=config.sample.noise_level,
            deterministic=deterministic,
            prev_sample=sample["next_latents"][:, j].float(),
        )
    return prev_sample_mean, std_dev_t

def compute_teacher_step_mean(
    transformer,
    pipeline,
    latents_at_j,
    next_latent_at_j,
    timestep_at_j,
    cond_embeds,
    cond_pooled,
    neg_embeds,
    neg_pooled,
    guidance_scale,
    noise_level,
    solver: str = "flow",
    deterministic: bool = False,
):

    if guidance_scale > 1.0:
        noise_pred = transformer(
            hidden_states=torch.cat([latents_at_j] * 2),
            timestep=torch.cat([timestep_at_j] * 2),
            encoder_hidden_states=torch.cat([neg_embeds, cond_embeds]),
            pooled_projections=torch.cat([neg_pooled, cond_pooled]),
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = transformer(
            hidden_states=latents_at_j,
            timestep=timestep_at_j,
            encoder_hidden_states=cond_embeds,
            pooled_projections=cond_pooled,
            return_dict=False,
        )[0]

    if solver == "flow" and not deterministic:
        _, _, prev_sample_mean, _ = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            timestep_at_j,
            latents_at_j.float(),
            prev_sample=next_latent_at_j.float(),
            noise_level=noise_level,
        )
    else:
        _, prev_sample_mean, _, _ = solver_step(
            scheduler=pipeline.scheduler,
            model_output=noise_pred.float(),
            timestep=timestep_at_j,
            sample=latents_at_j.float(),
            solver=solver,
            noise_level=noise_level,
            deterministic=deterministic,
            prev_sample=next_latent_at_j.float(),
        )
    return prev_sample_mean

def _eval_one_teacher(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    proxy_reward_fn,
    executor,
    autocast,
    log_prefix,
    save_subdir,
):
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    local_images = []
    local_captions = []
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc=f"Eval[{log_prefix or 'main'}]: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )

        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                eval_kw = dict(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    guidance_scale=config.sample.eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                    deterministic=True,
                    num_inference_steps=config.sample.eval_num_steps,
                )
                images, _, _ = pipeline_with_logprob(pipeline, **eval_kw)
            with torch.no_grad():
                primary_fut = executor.submit(
                    proxy_reward_fn, images, prompts, prompt_metadata, only_strict=False
                )
                time.sleep(0)
                proxy_rewards, _ = primary_fut.result()

        proc_id = accelerator.process_index
        for idx, i in enumerate(range(len(images))):
            image = images[i]
            pil = Image.fromarray(
                (image.detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((config.resolution, config.resolution))

            avg_proxy = proxy_rewards["avg"][i].item() if hasattr(proxy_rewards["avg"][i], "item") else float(proxy_rewards["avg"][i])
            caption = f"GPU:{proc_id} | {prompts[i]} | proxy: {avg_proxy:.2f}"

            local_images.append(pil)
            local_captions.append(caption)

        for key, value in proxy_rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).detach().float().cpu().numpy()
            all_rewards[f"proxy_{key}"].append(rewards_gather)

    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        all_images = [None] * world_size
        all_captions = [None] * world_size
        dist.all_gather_object(all_images, local_images)
        dist.all_gather_object(all_captions, local_captions)
        all_images = [item for sublist in all_images for item in sublist]
        all_captions = [item for sublist in all_captions for item in sublist]
    else:
        all_images = local_images
        all_captions = local_captions
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    if accelerator.is_main_process:
        num_samples = min(100, len(all_images))
        sample_indices = list(range(num_samples))
        sampled_images = [all_images[i] for i in sample_indices]
        sampled_captions = [all_captions[i] for i in sample_indices]
        eval_images = [wandb.Image(pil, caption=caption) for pil, caption in zip(sampled_images, sampled_captions)]
        for key, value in all_rewards.items():
            print(f"{log_prefix}{key}", value.shape)
        wandb.log(
            {
                f"eval_{log_prefix}images": eval_images,
                **{
                    f"eval_{log_prefix}monitor_{key}": np.mean(value[value != -10])
                    for key, value in all_rewards.items()
                },
                **{
                    f"eval_{log_prefix}monitor_{key}_std": np.std(value[value != -10])
                    for key, value in all_rewards.items()
                },
            },
            step=global_step,
        )
        save_eval_dir = getattr(config, "save_eval_dir", None)
        if save_eval_dir:
            save_dir = os.path.join(save_eval_dir, f"step_{global_step:04d}")
            if save_subdir:
                save_dir = os.path.join(save_dir, save_subdir)
            os.makedirs(save_dir, exist_ok=True)
            for i, image in enumerate(all_images[:num_samples]):
                image.save(os.path.join(save_dir, f"image_{i:04d}.png"))
    metrics: dict[str, float] = {}
    if accelerator.is_main_process:
        for key, value in all_rewards.items():
            if "_strict_accuracy" in key or "_accuracy" in key:
                continue
            valid = value[value != -10]
            if valid.size > 0:
                metrics[key] = float(np.mean(valid))
    return metrics

def eval(
    pipeline,
    test_dataloaders,
    teachers,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    proxy_reward_fns_per_teacher,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    use_prefix = len(teachers) > 1
    metrics_per_slot: dict[int, dict[str, float]] = {}
    for slot_idx, (tc, loader, proxy_reward_fn) in enumerate(
        zip(teachers, test_dataloaders, proxy_reward_fns_per_teacher)
    ):
        if proxy_reward_fn is None:

            continue
        log_prefix = f"{tc['name']}_" if use_prefix else ""
        save_subdir = tc["name"] if use_prefix else None
        slot_metrics = _eval_one_teacher(
            pipeline=pipeline,
            test_dataloader=loader,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            config=config,
            accelerator=accelerator,
            global_step=global_step,
            proxy_reward_fn=proxy_reward_fn,
            executor=executor,
            autocast=autocast,
            log_prefix=log_prefix,
            save_subdir=save_subdir,
        )
        metrics_per_slot[slot_idx] = slot_metrics

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

    return metrics_per_slot

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):

    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,

        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:

        accelerator.init_trackers(
            project_name="flow_grpo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device)
    teachers = _build_teachers_config(config)
    unique_adapter_lora = {}
    unique_adapter_guidance_scale = {}
    slot_adapter_names = []
    slot_adapter_guidance_scales = []

    def _register_adapter(name, lora_path, guidance_scale, slot_label):
        adapter = f"teacher_{name}"
        if adapter in unique_adapter_lora:
            if unique_adapter_lora[adapter] != lora_path:
                raise ValueError(
                    f"Teacher name '{name}' (slot '{slot_label}') "
                    f"references lora_path={lora_path!r} but the same "
                    f"name was previously registered with "
                    f"{unique_adapter_lora[adapter]!r}. Each "
                    f"`teacher_<name>` adapter must map to exactly one "
                    f"LoRA across the whole `config.train.teachers` "
                    f"list."
                )
            if unique_adapter_guidance_scale[adapter] != guidance_scale:
                raise ValueError(
                    f"Teacher name '{name}' (slot '{slot_label}') "
                    f"references guidance_scale={guidance_scale} but the "
                    f"same name was previously registered with "
                    f"guidance_scale="
                    f"{unique_adapter_guidance_scale[adapter]}. Each "
                    f"`teacher_<name>` adapter is one PEFT adapter on the "
                    f"shared backbone, so it must use exactly one CFG "
                    f"scale across the whole `config.train.teachers` "
                    f"list."
                )
        else:
            unique_adapter_lora[adapter] = lora_path
            unique_adapter_guidance_scale[adapter] = guidance_scale
        return adapter
    for tc in teachers:
        adapter_list = [
            _register_adapter(tc["name"], tc["lora_path"], tc["guidance_scale"], tc["name"])
        ]
        gs_list = [tc["guidance_scale"]]
        slot_adapter_names.append(adapter_list)
        slot_adapter_guidance_scales.append(gs_list)
    max_teachers_per_slot = max(len(a) for a in slot_adapter_names)
    if accelerator.is_local_main_process:
        logger.info(
            f"OPD: configured {len(teachers)} teacher slot(s) using "
            f"{len(unique_adapter_lora)} unique teacher adapter(s) "
            f"(K_max={max_teachers_per_slot}):"
        )
        for tc, adapters, gs_list in zip(
            teachers,
            slot_adapter_names,
            slot_adapter_guidance_scales,
        ):
            logger.info(
                f"  slot {tc['name']}: prompts={tc['dataset']}, "
                f"proxy={list(tc['proxy_reward'].keys())}, "
                f"teachers={adapters}, "
                f"guidance_scales={gs_list}"
            )
        for adapter, path in unique_adapter_lora.items():
            logger.info(
                f"  adapter {adapter} <- {path} "
                f"(guidance_scale={unique_adapter_guidance_scale[adapter]})"
            )

    original_reset = AcceleratorState._reset_state
    AcceleratorState._reset_state = lambda *args, **kwargs: None
    proxy_reward_fns_per_teacher = []
    for tc in teachers:
        if tc["proxy_reward"]:
            fn = getattr(flow_grpo.rewards, "multi_score")(
                accelerator.device, tc["proxy_reward"]
            )
            fn = make_no_grad(fn)
        else:

            fn = None
        proxy_reward_fns_per_teacher.append(fn)

    AcceleratorState._reset_state = original_reset
    torch.cuda.empty_cache()

    if config.use_lora:

        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)

            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

        for adapter_name, lora_path in unique_adapter_lora.items():
            pipeline.transformer.load_adapter(lora_path, adapter_name=adapter_name)

        for n, p in pipeline.transformer.named_parameters():
            if "teacher_" in n:
                p.requires_grad = False

        pipeline.transformer.set_adapter("default")

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    def _resolve_dataset_cls(prompt_fn):
        if prompt_fn == "general_ocr":
            return TextPromptDataset
        elif prompt_fn == "geneval":
            return GenevalPromptDataset
        else:
            raise NotImplementedError(
                "Only `general_ocr` and `geneval` prompt_fns are supported."
            )

    teacher_dataset_classes = [
        _resolve_dataset_cls(tc["prompt_fn"]) for tc in teachers
    ]

    train_datasets = [
        cls(tc["dataset"], "train")
        for cls, tc in zip(teacher_dataset_classes, teachers)
    ]
    test_datasets = [
        cls(tc["dataset"], "test")
        for cls, tc in zip(teacher_dataset_classes, teachers)
    ]

    train_samplers = []
    train_dataloaders = []
    for cls, ds in zip(teacher_dataset_classes, train_datasets):
        sampler = DistributedKRepeatSampler(
            dataset=ds,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=1,
            collate_fn=cls.collate_fn,

        )
        train_samplers.append(sampler)
        train_dataloaders.append(loader)

    test_dataloaders = [
        DataLoader(
            ds,
            batch_size=config.sample.test_batch_size,
            collate_fn=cls.collate_fn,
            shuffle=False,
            num_workers=8,
        )
        for cls, ds in zip(teacher_dataset_classes, test_datasets)
    ]

    if len(teachers) > 1:
        assert config.sample.num_batches_per_epoch % len(teachers) == 0, (
            f"`config.sample.num_batches_per_epoch` "
            f"(={config.sample.num_batches_per_epoch}) must be divisible by "
            f"the number of teachers (={len(teachers)}) so each teacher gets "
            f"the same number of sampling batches per epoch."
        )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    prepared = accelerator.prepare(
        transformer,
        optimizer,
        *train_dataloaders,
        *test_dataloaders,
    )
    n_t = len(teachers)
    transformer = prepared[0]
    optimizer = prepared[1]
    train_dataloaders = list(prepared[2 : 2 + n_t])
    test_dataloaders = list(prepared[2 + n_t : 2 + 2 * n_t])

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0

    train_iters = [iter(loader) for loader in train_dataloaders]

    slot_idx_to_name = [tc["name"] for tc in teachers]

    slot_idx_to_teacher_short_names = [
        [name.replace("teacher_", "", 1) for name in adapters]
        for adapters in slot_adapter_names
    ]

    while True:

        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval_metrics_per_slot = eval(
                pipeline,
                test_dataloaders,
                teachers,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                global_step,
                proxy_reward_fns_per_teacher,
                executor,
                autocast,
                num_train_timesteps,
                ema,
                transformer_trainable_parameters,
            )

        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):

            teacher_idx = i % len(teachers)
            slot_adapters = slot_adapter_names[teacher_idx]
            slot_guidance_scales = slot_adapter_guidance_scales[teacher_idx]
            train_samplers[teacher_idx].set_epoch(
                epoch * config.sample.num_batches_per_epoch + i
            )
            prompts, prompt_metadata = next(train_iters[teacher_idx])

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():

                    sample_solver, sample_deterministic = _resolve_solver(config)
                    sample_kw = dict(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        solver=sample_solver,
                        deterministic=sample_deterministic,
                    )
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        **sample_kw,
                    )

            latents = torch.stack(
                latents, dim=1
            )
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )
            if timesteps.shape[0] != latents.shape[0]:
                timesteps = timesteps[: latents.shape[0]]

            sample_neg_prompt_embeds_b = sample_neg_prompt_embeds[: prompt_embeds.shape[0]]
            sample_neg_pooled_prompt_embeds_b = sample_neg_pooled_prompt_embeds[
                : pooled_prompt_embeds.shape[0]
            ]

            slot_teacher_means_per_teacher = []
            with autocast():
                with torch.no_grad():
                    for k_idx, adapter_name in enumerate(slot_adapters):
                        unwrap_model(transformer, accelerator).set_adapter(adapter_name)
                        with contextlib.nullcontext():

                            teacher_gs = slot_guidance_scales[k_idx]
                            teacher_prev_sample_mean_list = []
                            for j in range(config.sample.num_steps):
                                teacher_prev_mean_j = compute_teacher_step_mean(
                                        transformer=transformer,
                                        pipeline=pipeline,
                                        latents_at_j=latents[:, j],
                                        next_latent_at_j=latents[:, j + 1],
                                        timestep_at_j=timesteps[:, j],
                                        cond_embeds=prompt_embeds,
                                        cond_pooled=pooled_prompt_embeds,
                                        neg_embeds=sample_neg_prompt_embeds_b,
                                        neg_pooled=sample_neg_pooled_prompt_embeds_b,
                                        guidance_scale=teacher_gs,
                                        noise_level=config.sample.noise_level,
                                        solver=sample_solver,
                                        deterministic=sample_deterministic,
                                    )
                                teacher_prev_sample_mean_list.append(
                                    teacher_prev_mean_j
                                )
                            slot_teacher_means_per_teacher.append(
                                torch.stack(
                                    teacher_prev_sample_mean_list, dim=1
                                ).float()
                            )

            unwrap_model(transformer, accelerator).set_adapter("default")

            teacher_prev_sample_means_slot = torch.stack(
                slot_teacher_means_per_teacher, dim=2
            )
            slot_K = teacher_prev_sample_means_slot.shape[2]

            if slot_K < max_teachers_per_slot:
                B_dim, T_dim, _, C_dim, H_dim, W_dim = (
                    teacher_prev_sample_means_slot.shape
                )
                pad = torch.zeros(
                    B_dim,
                    T_dim,
                    max_teachers_per_slot - slot_K,
                    C_dim,
                    H_dim,
                    W_dim,
                    dtype=teacher_prev_sample_means_slot.dtype,
                    device=teacher_prev_sample_means_slot.device,
                )
                teacher_prev_sample_means = torch.cat(
                    [teacher_prev_sample_means_slot, pad], dim=2
                )
            else:
                teacher_prev_sample_means = teacher_prev_sample_means_slot

            batch_proxy_fn = proxy_reward_fns_per_teacher[teacher_idx]
            if batch_proxy_fn is None:
                monitor_fut = None
            else:
                with torch.no_grad():
                    monitor_fut = executor.submit(
                        batch_proxy_fn,
                        images,
                        prompts,
                        prompt_metadata,
                        only_strict=True,
                    )
                time.sleep(0)

            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],
                    "next_latents": latents[
                        :, 1:
                    ],

                    "teacher_prev_sample_means": teacher_prev_sample_means,
                    "monitor": monitor_fut,
                }
            )

        for sample in tqdm(
            samples,
            desc="Waiting for monitor scores",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            if sample["monitor"] is None:
                sample["monitor"] = {}
            else:
                with torch.no_grad():
                    proxy_out, _proxy_meta = sample["monitor"].result()
                sample["monitor"] = {
                    key: torch.as_tensor(value, device=accelerator.device).float()
                    for key, value in proxy_out.items()
                }

        all_monitor_keys = sorted({k for s in samples for k in s["monitor"].keys()})
        for s in samples:
            n = s["timesteps"].shape[0]
            for k in all_monitor_keys:
                if k not in s["monitor"]:
                    s["monitor"][k] = torch.full(
                        (n,), -10.0, device=accelerator.device, dtype=torch.float32
                    )

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:

            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]

                if "avg" in samples["monitor"]:
                    mon_avg = samples["monitor"]["avg"][-len(images) :]
                    sampled_monitor = [mon_avg[i].item() for i in sample_indices]
                else:
                    sampled_monitor = [float("nan")] * len(sample_indices)
                def _format_caption(prompt, proxy_avg):
                    return f"{prompt:.100} | proxy_avg: {float(proxy_avg):.2f}"

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=_format_caption(prompt, proxy_avg),
                            )
                            for idx, (prompt, proxy_avg) in enumerate(
                                zip(sampled_prompts, sampled_monitor)
                            )
                        ],
                    },
                    step=global_step,
                )

        gathered_monitor = {
            key: accelerator.gather(value).detach().float().cpu().numpy()
            for key, value in samples["monitor"].items()
        }
        if accelerator.is_main_process:
            monitor_log = {"epoch": epoch}
            for key, value in gathered_monitor.items():
                if "_strict_accuracy" in key or "_accuracy" in key:
                    continue
                valid = value[value != -10]
                if valid.size > 0:
                    monitor_log[f"monitor_{key}"] = float(valid.mean())
            wandb.log(monitor_log, step=global_step)

        del samples["monitor"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps

        for inner_epoch in range(config.train.num_inner_epochs):

            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:

                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                                prev_sample_mean, std_dev_t = compute_student_step(
                                    transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )

                        teacher_prev_mean = sample[
                            "teacher_prev_sample_means"
                        ][:, j].detach()

                        delta = (
                            prev_sample_mean.float().unsqueeze(1)
                            - teacher_prev_mean.float()
                        )

                        if float(config.sample.noise_level) <= 0.0:
                            per_step_kl = 0.5 * (delta ** 2)
                        else:

                            sigma_sq = (
                                std_dev_t.float() ** 2
                            ).clamp(min=1e-8).unsqueeze(1)
                            per_step_kl = (delta ** 2) / (2.0 * sigma_sq)

                        per_step_kl_per_teacher_scalar = per_step_kl.mean(
                            dim=tuple(range(2, per_step_kl.ndim))
                        )
                        per_step_kl_scalar = per_step_kl_per_teacher_scalar.sum(dim=1)

                        distill_loss = per_step_kl_scalar.mean()
                        loss = distill_loss

                        info["policy_loss"].append(distill_loss)
                        info["train_reverse_kl"].append(per_step_kl_scalar.mean())
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                            info["grad_norm"].append(grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:

                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch+=1

if __name__ == "__main__":
    app.run(main)
