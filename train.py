import logging
import os
import random
import re
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from transformers.trainer_utils import get_last_checkpoint

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb
    from omegaconf import OmegaConf

    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    wandb_run = wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


def get_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    total_devices = get_total_devices()

    div = per_device_train_batch_size*total_devices
    steps = train_batch_size/div
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True

    if OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        accumulation_steps = compute_accumulation_steps(
            train_batch_size=cfg.train_batch_size,
            per_device_train_batch_size=cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps = accumulation_steps

    logger.info(f"Accumulation steps {cfg.gradient_accumulation_steps} ----")

    using_wandb = False
    if isinstance(cfg.report_to, str):
        using_wandb = cfg.report_to == 'wandb'
    elif cfg.report_to is not None:
        for v in cfg.report_to:
            using_wandb = using_wandb or (v == 'wandb')

    if using_wandb and is_main_process:
        wandb = wandb_init(
            cfg=cfg,
            group_name=cfg.wandb_group_name,
            run_name=cfg.wandb_run_name,
            log_dir=cfg.output_dir,
        )

    tokenizer = hydra.utils.instantiate(cfg.make_tokenizer_fn)

    datasets = hydra.utils.instantiate(
        cfg.make_dataset_fn, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        **datasets,
    )

    print('Model initialized!!!')

    last_checkpoint = get_checkpoint(cfg.output_dir)
    if not last_checkpoint and cfg.resume_from is not None:
        last_checkpoint = get_checkpoint(cfg.resume_from)
    if last_checkpoint:
        logger.info("Found checkpoint, resuming training run from "
                    f"{last_checkpoint}.")
    else:
        logger.info("No existing checkpoint, initializing new model")

    logger.info(f"Training  {datetime.now()}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info(f"Training complete {datetime.now()}")

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if cfg.save_final_model:
        logger.info(f"Saving final model at {cfg.output_dir}")
        trainer.model.config.use_cache = True
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        logger.info(f"Done saving {datetime.now()}")

    if is_main_process and cfg.push_to_hub:
        tags = cfg.tags if cfg.tags is not None else []
        trainer.create_model_card({"tags": tags})
    if cfg.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    if is_main_process and cfg.call_post_training is not None:

        hydra.utils.instantiate(cfg.call_post_training)


if __name__ == "__main__":
    main()
