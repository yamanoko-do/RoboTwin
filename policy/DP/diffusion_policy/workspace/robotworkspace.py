if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import datetime
import shutil
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random, time, collections
import subprocess
import sys
import swanlab as wandb
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Set device early so model instantiation lands on the correct GPU
        _local_rank = os.environ.get("LOCAL_RANK")
        if _local_rank is not None:
            torch.cuda.set_device(int(_local_rank))

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # DDP initialization
        dist_mode = cfg.training.get("dist_mode", False)
        if dist_mode:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=3))
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            local_rank = 0
            global_rank = 0
            world_size = 1
            device = torch.device(cfg.training.get("device", "cuda:0"))

        # resume training (before DDP wrapping)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(
            dataset, **cfg.dataloader,
            dist_mode=dist_mode, rank=local_rank, world_size=world_size,
        )
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(
            val_dataset, **cfg.val_dataloader,
            dist_mode=dist_mode, rank=local_rank, world_size=world_size,
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # lr_total_epochs allows the scheduler to span the full training
        # even when num_epochs is set to a shorter round for train-eval loops
        lr_total_epochs = cfg.training.get("lr_total_epochs", cfg.training.num_epochs)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * lr_total_epochs) //
            cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None

        # configure logging (rank 0 only)
        if global_rank == 0:
            logging_kwargs = {k: v for k, v in cfg.logging.items() if k != 'mode'}
            wandb_run = wandb.init(
                workspace="limxooo",
                logdir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.logging.get('mode', 'disabled'),
                **logging_kwargs
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
            # Persist run id so train-eval loop can resume the same run
            id_file = os.path.join(str(self.output_dir), ".swanlab_run_id")
            with open(id_file, "w") as f:
                f.write(wandb_run.id)
        else:
            wandb_run = None

        # configure checkpoint
        if global_rank == 0:
            topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                                 **cfg.checkpoint.topk)
        if dist_mode:
            dist.barrier()

        # device transfer and DDP wrapping
        self.model.to(device)
        if dist_mode:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.eval_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        # Only rank 0 writes to the json log to avoid corruption in DDP
        json_logger = JsonLogger(log_path) if global_rank == 0 else None
        if json_logger is not None:
            json_logger.start()

        try:
            training_start_time = time.time()
            recent_iter_times = collections.deque(maxlen=20)  # sliding window for stable ETA
            iter_start = time.time()
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if dist_mode:
                    train_dataloader.sampler.set_epoch(self.epoch)

                # freeze LingBotDepth backbone only (unwrap DDP to access obs_encoder)
                # ResNet and LingBotDepth projection head remain trainable.
                if cfg.training.freeze_encoder:
                    model_for_freeze = self.model.module if dist_mode else self.model
                    for name, param in model_for_freeze.obs_encoder.named_parameters():
                        # Freeze only the DINOv2 ViT backbone inside LingBotDepth;
                        # keep ResNet, projection heads, and other modules trainable.
                        if 'mdm_model' in name:
                            param.requires_grad = False

                train_losses = list()
                total_epochs = cfg.training.get("lr_total_epochs", cfg.training.num_epochs)
                iters_per_epoch = len(train_dataloader)
                total_iters = total_epochs * iters_per_epoch
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=(global_rank != 0),
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss (through forward() for DDP gradient sync)
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema (use unwrapped model)
                        if cfg.training.use_ema:
                            ema.step(self.model.module if dist_mode else self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        # ETA estimation (global: remaining iters across all epochs)
                        recent_iter_times.append(time.time() - iter_start)
                        iter_start = time.time()
                        round_done_iters = batch_idx + 1
                        global_done_iters = self.epoch * iters_per_epoch + round_done_iters
                        global_remaining_iters = total_iters - global_done_iters
                        training_elapsed = time.time() - training_start_time
                        avg_iter_sec = sum(recent_iter_times) / len(recent_iter_times)
                        remaining_sec = avg_iter_sec * max(global_remaining_iters, 0)
                        time_cost = tqdm.tqdm.format_interval(training_elapsed)
                        time_left = tqdm.tqdm.format_interval(remaining_sec)
                        tepoch.set_postfix(
                            loss=raw_loss_cpu,
                            time_cost=time_cost,
                            ETA=time_left,
                            refresh=False,
                        )
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train/loss": raw_loss_cpu,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        }

                        # per-iteration logging (rank 0 only)
                        if self.global_step % cfg.training.log_every == 0 and global_rank == 0:
                            wandb.log(step_log, step=self.global_step)

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            if global_rank == 0:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                train_loss = np.mean(train_losses)
                step_log["train/loss"] = train_loss

                # increment epoch before checkpoint so saved value is correct
                self.epoch += 1

                # --- Checkpoint (rank 0 only) ---
                do_ckpt = (self.epoch % cfg.training.checkpoint_every) == 0
                if do_ckpt and global_rank == 0:
                    if self._saving_thread is not None and self._saving_thread.is_alive():
                        self._saving_thread.join()
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    ckpt_path = pathlib.Path(self.output_dir).joinpath(
                        "checkpoints", f"{save_name}-{seed}", f"{self.epoch}.ckpt")
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(path=ckpt_path, use_thread=False)
                    # Also save as latest.ckpt so resume can find it
                    latest_path = self.get_checkpoint_path()
                    shutil.copy2(str(ckpt_path), str(latest_path))
                    print(f"\033[93m[Checkpoint] Saved to {ckpt_path} (latest -> {latest_path})\033[0m")

                # Synchronize all ranks after checkpoint
                if dist_mode:
                    dist.barrier()

                # end of epoch
                if global_rank == 0:
                    json_logger.log(step_log)
                if self.global_step % cfg.training.log_every != 0 and global_rank == 0:
                    wandb.log(step_log, step=self.global_step)
                self.global_step += 1
        finally:
            if json_logger is not None:
                json_logger.stop()


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


class DistributedBatchSampler:
    """Partitions samples across DDP ranks, then forms per-rank batches.

    Unlike the old approach (which split pre-formed batches across ranks),
    this sampler first divides the full sample index set evenly among ranks,
    then each rank reshapes its share into batches of per_rank_batch_size.
    This guarantees every rank yields the same number of batches.
    """

    def __init__(self, data_size: int, global_batch_size: int,
                 num_replicas: int, rank: int,
                 shuffle: bool = False, seed: int = 0):
        self.data_size = data_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Drop samples so data_size is divisible by num_replicas
        self.per_rank_samples = data_size // num_replicas
        # Per-rank batch size (global_batch_size / num_replicas)
        assert global_batch_size % num_replicas == 0, \
            f"global_batch_size ({global_batch_size}) must be divisible by world_size ({num_replicas})"
        self.per_rank_batch_size = global_batch_size // num_replicas
        self.num_batch = self.per_rank_samples // self.per_rank_batch_size
        self.rng = np.random.default_rng(seed) if shuffle else None

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)

        # Drop tail so total is divisible by num_replicas
        usable = self.per_rank_samples * self.num_replicas
        perm = perm[:usable]

        # Each rank takes a contiguous chunk (strided partitioning like DistributedSampler)
        rank_indices = perm[self.rank::self.num_replicas]

        # Drop tail so rank's share is divisible by per_rank_batch_size
        usable_rank = self.num_batch * self.per_rank_batch_size
        rank_indices = rank_indices[:usable_rank]

        # Reshape into batches
        rank_indices = rank_indices.reshape(self.num_batch, self.per_rank_batch_size)
        for i in range(self.num_batch):
            yield rank_indices[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
    dist_mode: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    def collate(x):
        assert len(x) == 1
        return x[0]

    if dist_mode:
        # batch_size from config is the GLOBAL batch size.
        # Each rank processes batch_size/world_size samples per step.
        assert batch_size % world_size == 0, \
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        per_rank_batch_size = batch_size // world_size
        # Dataset pre-allocates buffers sized for self.batch_size.
        # In DDP each rank only sees per_rank_batch_size samples, so resize.
        if dataset.batch_size != per_rank_batch_size:
            dataset._resize_buffers(per_rank_batch_size)
        batch_sampler = DistributedBatchSampler(
            data_size=len(dataset),
            global_batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
