import importlib.util
import os

import ml_collections

def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def mopd():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.teacher_guidance_scale = 4.5

    config.resolution = 512
    config.run_project = "flow_grpo"

    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 3
    assert config.sample.num_batches_per_epoch % 3 == 0, (
        "Please set config.sample.num_batches_per_epoch to a multiple of "
        "len(config.train.teachers) (=3 here)."
    )
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"

    config.sample.noise_level = 0.0

    config.save_freq = 30
    config.eval_freq = 30

    config.reward_fn = ml_collections.ConfigDict()
    config.proxy_reward_fn = {}
    config.prompt_fn = "general_ocr"

    pickscore_lora = "YOUR_AES_TEACHER_PATH"
    pickscore_teacher_gs = 4.5
    ocr_lora = "YOUR_OCR_TEACHER_PATH"
    ocr_teacher_gs = 4.5
    geneval_lora = "YOUR_GENEVAL_TEACHER_PATH"
    geneval_teacher_gs = 1.0
    config.train.teachers = [
        {
            "name": "pickscore",
            "dataset": os.path.join(os.getcwd(), "dataset/pickscore"),
            "lora_path": pickscore_lora,
            "proxy_reward": {"pickscore": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": pickscore_teacher_gs,
        },
        {
            "name": "ocr",
            "dataset": os.path.join(os.getcwd(), "dataset/ocr"),
            "lora_path": ocr_lora,
            "proxy_reward": {"ocr": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": ocr_teacher_gs,
        },
        {
            "name": "geneval",
            "dataset": os.path.join(os.getcwd(), "dataset/geneval"),
            "lora_path": geneval_lora,
            "proxy_reward": {"geneval": 1.0},
            "prompt_fn": "geneval",
            "guidance_scale": geneval_teacher_gs,
        },
    ]
    config.train.beta = 0.0

    config.run_name = "[MOPD + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def sopd_base():
    """Shared single-teacher OPD defaults for SD3.5-M.

    Mirrors `mopd()` but without the multi-teacher-specific assertion;
    each `sopd_*` wrapper only needs to set the dataset, prompt_fn,
    a single `config.train.teachers` entry, and the run/save name.
    """
    config = compressibility()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.teacher_guidance_scale = 4.5

    config.resolution = 512
    config.run_project = "flow_grpo"

    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 3
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"

    config.sample.noise_level = 0.0

    config.save_freq = 30
    config.eval_freq = 30

    config.reward_fn = ml_collections.ConfigDict()
    config.proxy_reward_fn = {}
    config.train.beta = 0.0

    return config


def sopd_pickscore():
    config = sopd_base()

    pickscore_lora = "YOUR_AES_TEACHER_PATH"
    pickscore_teacher_gs = 4.5

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.train.teachers = [
        {
            "name": "pickscore",
            "dataset": os.path.join(os.getcwd(), "dataset/pickscore"),
            "lora_path": pickscore_lora,
            "proxy_reward": {"pickscore": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": pickscore_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def sopd_ocr():
    config = sopd_base()

    ocr_lora = "YOUR_OCR_TEACHER_PATH"
    ocr_teacher_gs = 4.5

    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.prompt_fn = "general_ocr"
    config.train.teachers = [
        {
            "name": "ocr",
            "dataset": os.path.join(os.getcwd(), "dataset/ocr"),
            "lora_path": ocr_lora,
            "proxy_reward": {"ocr": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": ocr_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + OCR]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def sopd_geneval():
    config = sopd_base()

    geneval_lora = "YOUR_GENEVAL_TEACHER_PATH"
    geneval_teacher_gs = 1.0

    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.prompt_fn = "geneval"
    config.train.teachers = [
        {
            "name": "geneval",
            "dataset": os.path.join(os.getcwd(), "dataset/geneval"),
            "lora_path": geneval_lora,
            "proxy_reward": {"geneval": 1.0},
            "prompt_fn": "geneval",
            "guidance_scale": geneval_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def get_config(name):
    return globals()[name]()
