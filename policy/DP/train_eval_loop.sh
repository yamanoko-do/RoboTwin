#!/bin/bash
# train_eval_loop.sh - Alternating train N epochs, evaluate, repeat.
#
# Training resumes from checkpoint automatically (lr, optimizer state, epoch
# counter are all restored). Each round writes latest.ckpt so the next round
# picks up exactly where the previous one left off.
#
# Evaluation runs via eval_checkpoint.py across all GPUs in parallel.
# After all eval processes finish, results are aggregated and logged
# (metrics + video) to the same SwanLab run as training.
#
# Usage:
#   bash train_eval_loop.sh cfg=train_cfg.yaml
#
#   All parameters (task, data path, training hyperparams) live in train_cfg.yaml.
#   CLI overrides still work (highest priority):
#     bash train_eval_loop.sh cfg=train_cfg.yaml epochs=1000 device=0,1,2,3
#
# Key-value options (override train_cfg.yaml):
#   task_name=NAME         task name (e.g. move_playingcard_away)
#   task_config=NAME       task config (e.g. demo_clean)
#   zarr_path=PATH         path to training data
#   name=NAME              output directory name
#   epochs=N               total training epochs
#   eval_every=N           train N epochs before each eval round
#   device=GPUS            comma-separated GPU ids
#   cfg=PATH               path to config yaml (default: train_cfg.yaml)

set -euo pipefail

# ======================== arguments ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default config file
CFG_FILE="${SCRIPT_DIR}/train_cfg.yaml"

# Parse key=value options (can override config values)
for arg in "$@"; do
    case "${arg}" in
        task_name=*)   _OPT_TASK_NAME="${arg#task_name=}" ;;
        task_config=*) _OPT_TASK_CONFIG="${arg#task_config=}" ;;
        zarr_path=*)   _OPT_ZARR_PATH="${arg#zarr_path=}" ;;
        name=*)        _OPT_NAME="${arg#name=}" ;;
        epochs=*)      _OPT_EPOCHS="${arg#epochs=}" ;;
        eval_every=*)  _OPT_EVAL_EVERY="${arg#eval_every=}" ;;
        device=*)      _OPT_DEVICE="${arg#device=}" ;;
        cfg=*)         CFG_FILE="${arg#cfg=}" ;;
        *)             echo "Unknown option: ${arg}"; exit 1 ;;
    esac
done

# ======================== load config from YAML ========================
# Helper: read a value from the cfg file using python (handles nested keys like pipeline.total_epochs)
_cfg_val() {
    local key="$1"
    python3 -c "
import yaml
with open('${CFG_FILE}') as f:
    cfg = yaml.safe_load(f)
keys = '${key}'.split('.')
v = cfg
for k in keys:
    v = v[k]
# Format lists as Hydra-compatible [a, b, c]
if isinstance(v, list):
    print('[' + ', '.join(str(x) for x in v) + ']')
elif v is None:
    print('null')
else:
    print(v)
"
}

# Task params (from cfg, then CLI override)
TASK_NAME="${_OPT_TASK_NAME:-$(_cfg_val task.name)}"
TASK_CONFIG="${_OPT_TASK_CONFIG:-$(_cfg_val task.config)}"
ZARR_PATH="${_OPT_ZARR_PATH:-$(_cfg_val task.zarr_path)}"
HYDRA_CONFIG_NAME=$(_cfg_val hydra.config_name)

# Pipeline params (from cfg, then CLI override)
TOTAL_EPOCHS="${_OPT_EPOCHS:-$(_cfg_val pipeline.total_epochs)}"
EPOCHS_PER_ROUND="${_OPT_EVAL_EVERY:-$(_cfg_val pipeline.eval_every)}"
GPU_IDS="${_OPT_DEVICE:-$(_cfg_val pipeline.device)}"
SEED=$(_cfg_val pipeline.seed)
HEAD_CAMERA_TYPE=$(_cfg_val pipeline.head_camera_type)
TOTAL_EVAL_EPISODES=$(_cfg_val pipeline.eval_episodes)
EXPERIMENT_NAME="${_OPT_NAME:-$(_cfg_val pipeline.experiment_name)}"

# Training params (from cfg, used to build Hydra overrides)
_cfg_lr_total_epochs=$(_cfg_val training.lr_total_epochs)
_cfg_training_dist_mode=$(_cfg_val training.dist_mode)
_cfg_training_resume=$(_cfg_val training.resume)
_cfg_training_checkpoint_every=$(_cfg_val training.checkpoint_every)
_cfg_training_use_ema=$(_cfg_val training.use_ema)
_cfg_training_freeze_encoder=$(_cfg_val training.freeze_encoder)
_cfg_training_lr_scheduler=$(_cfg_val training.lr_scheduler)
_cfg_training_lr_warmup_steps=$(_cfg_val training.lr_warmup_steps)
_cfg_training_gradient_accumulate_every=$(_cfg_val training.gradient_accumulate_every)
_cfg_training_eval_every=$(_cfg_val training.eval_every)
_cfg_training_eval_episodes=$(_cfg_val training.eval_episodes)
_cfg_training_val_every=$(_cfg_val training.val_every)
_cfg_training_sample_every=$(_cfg_val training.sample_every)
_cfg_training_log_every=$(_cfg_val training.log_every)
_cfg_training_seed=$(_cfg_val training.seed)
_cfg_training_debug=$(_cfg_val training.debug)

# Optimizer
_cfg_optimizer_lr=$(_cfg_val optimizer.lr)
_cfg_optimizer_betas=$(_cfg_val optimizer.betas)
_cfg_optimizer_eps=$(_cfg_val optimizer.eps)
_cfg_optimizer_weight_decay=$(_cfg_val optimizer.weight_decay)

# Dataloader
_cfg_dataloader_batch_size=$(_cfg_val dataloader.batch_size)
_cfg_dataloader_num_workers=$(_cfg_val dataloader.num_workers)
_cfg_dataloader_shuffle=$(_cfg_val dataloader.shuffle)
_cfg_dataloader_pin_memory=$(_cfg_val dataloader.pin_memory)
_cfg_dataloader_persistent_workers=$(_cfg_val dataloader.persistent_workers)

_cfg_val_dataloader_batch_size=$(_cfg_val val_dataloader.batch_size)
_cfg_val_dataloader_num_workers=$(_cfg_val val_dataloader.num_workers)
_cfg_val_dataloader_shuffle=$(_cfg_val val_dataloader.shuffle)
_cfg_val_dataloader_pin_memory=$(_cfg_val val_dataloader.pin_memory)
_cfg_val_dataloader_persistent_workers=$(_cfg_val val_dataloader.persistent_workers)

# Policy
_cfg_policy_horizon=$(_cfg_val policy.horizon)
_cfg_policy_n_obs_steps=$(_cfg_val policy.n_obs_steps)
_cfg_policy_n_action_steps=$(_cfg_val policy.n_action_steps)
_cfg_policy_n_latency_steps=$(_cfg_val policy.n_latency_steps)
_cfg_policy_obs_as_global_cond=$(_cfg_val policy.obs_as_global_cond)
_cfg_policy_num_inference_steps=$(_cfg_val policy.num_inference_steps)
_cfg_policy_diffusion_step_embed_dim=$(_cfg_val policy.diffusion_step_embed_dim)
_cfg_policy_down_dims=$(_cfg_val policy.down_dims)
_cfg_policy_kernel_size=$(_cfg_val policy.kernel_size)
_cfg_policy_n_groups=$(_cfg_val policy.n_groups)
_cfg_policy_cond_predict_scale=$(_cfg_val policy.cond_predict_scale)

# Dataset
_cfg_dataset_val_ratio=$(_cfg_val dataset.val_ratio)
_cfg_dataset_max_train_episodes=$(_cfg_val dataset.max_train_episodes)

# Logging
_cfg_logging_project=$(_cfg_val logging.project)
_cfg_logging_mode=$(_cfg_val logging.mode)
_cfg_logging_resume=$(_cfg_val logging.resume)

# Eval
_cfg_eval_instruction_type=$(_cfg_val eval.instruction_type)

# Derived
NUM_TRAIN_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
POLICY_NAME=DP
IFS=',' read -ra EVAL_GPUS <<< "${GPU_IDS}"
NUM_EVAL_GPUS=${#EVAL_GPUS[@]}

# ======================== directories ========================
if [[ -n "${EXPERIMENT_NAME}" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/data/outputs/${EXPERIMENT_NAME}"
else
    OUTPUT_DIR="${SCRIPT_DIR}/data/outputs/train_eval_${TASK_NAME}_${TASK_CONFIG}"
fi
mkdir -p "${OUTPUT_DIR}"

# ======================== SwanLab run persistence ========================
SWANLAB_ID_FILE="${OUTPUT_DIR}/.swanlab_run_id"

# ======================== helpers ========================
log()  { echo -e "\033[96m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[93m[$(date +%H:%M:%S)] WARNING: $*\033[0m"; }
err()  { echo -e "\033[91m[$(date +%H:%M:%S)] ERROR: $*\033[0m" >&2; }

# Find the newest epoch checkpoint under OUTPUT_DIR
find_latest_ckpt() {
    # Sort by the numeric epoch in the filename (e.g. 50.ckpt -> 50)
    find "${OUTPUT_DIR}/checkpoints" -name "*.ckpt" ! -name "latest.ckpt" 2>/dev/null \
        | perl -e 'print sort { ($a =~ m{/(\d+)\.ckpt$})[0] <=> ($b =~ m{/(\d+)\.ckpt$})[0] } <>' \
        | tail -1
}

current_epoch() {
    local ckpt=$(find_latest_ckpt)
    if [[ -z "${ckpt}" ]]; then
        echo 0
    else
        basename "${ckpt}" .ckpt
    fi
}

# ======================== eval ========================
run_eval() {
    local ckpt_path=$1
    local epoch_num=$2

    log "Running evaluation: ${TOTAL_EVAL_EPISODES} episodes across ${NUM_EVAL_GPUS} GPUs, epoch ${epoch_num}"

    local eval_script="${SCRIPT_DIR}/diffusion_policy/evaluation/eval_checkpoint.py"
    local eval_log_dir="${OUTPUT_DIR}/eval_epoch${epoch_num}"
    mkdir -p "${eval_log_dir}"

    # Launch eval processes in parallel — one per eval GPU
    local pids=()
    local gpu_idx=0
    for gpu_id in "${EVAL_GPUS[@]}"; do
        local gpu_eval_log="${eval_log_dir}/gpu${gpu_id}.log"
        log "  GPU ${gpu_id}: rank ${gpu_idx}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python "${eval_script}" \
            --checkpoint "${ckpt_path}" \
            --eval_task_name "${TASK_NAME}" \
            --eval_task_config "${TASK_CONFIG}" \
            --head_camera_type "${HEAD_CAMERA_TYPE}" \
            --num_episodes "${TOTAL_EVAL_EPISODES}" \
            --instruction_type "${_cfg_eval_instruction_type}" \
            --output_dir "${OUTPUT_DIR}" \
            --epoch "${epoch_num}" \
            --rank "${gpu_idx}" \
            --world_size "${NUM_EVAL_GPUS}" \
        > "${gpu_eval_log}" 2>&1 &
        pids+=($!)
        gpu_idx=$(( gpu_idx + 1 ))
    done

    # Wait for all eval processes
    local all_ok=true
    for i in "${!pids[@]}"; do
        if ! wait ${pids[$i]}; then
            err "Eval process ${pids[$i]} (GPU ${EVAL_GPUS[$i]}) failed"
            all_ok=false
        fi
    done

    # Aggregate results from per-rank JSON files
    python3 -c "
import json, os, glob, sys

epoch = ${epoch_num}
output_dir = '${OUTPUT_DIR}'
swanlab_id_file = '${SWANLAB_ID_FILE}'

# Collect per-rank results
total_success = 0
total_episodes = 0
best_video = None  # prefer success video

rank_files = sorted(glob.glob(os.path.join(output_dir, f'eval_epoch{epoch}_rank*.json')))
for rf in rank_files:
    with open(rf) as f:
        d = json.load(f)
    total_success += d['success_count']
    total_episodes += d['total_episodes']
    # Pick first success video, or first video if none succeeded
    vp = d.get('video_path')
    if vp and os.path.exists(vp):
        if best_video is None:
            best_video = vp
        elif d['success_count'] > 0 and best_video is not None:
            # Replace with a success video if we only had a fail so far
            try:
                with open(glob.glob(os.path.join(output_dir, f'eval_epoch{epoch}_rank*.json'))[0]) as ff:
                    prev = json.load(ff)
                if prev.get('success_count', 0) == 0:
                    best_video = vp
            except:
                pass
    # Clean up rank file
    os.remove(rf)

success_rate = total_success / max(total_episodes, 1)
print(f'\033[96m[Eval] Epoch {epoch}: {total_success}/{total_episodes} = {success_rate:.1%}\033[0m')

# Save combined result
result_path = os.path.join(output_dir, f'eval_epoch{epoch}.json')
with open(result_path, 'w') as f:
    json.dump({
        'epoch': epoch,
        'success_rate': success_rate,
        'success_count': total_success,
        'total_episodes': total_episodes,
    }, f)

# Log to SwanLab
try:
    import swanlab

    # Load cfg for logging mode
    import torch, dill
    payload = torch.load(open('${ckpt_path}', 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)

    init_kwargs = dict(
        workspace='limxooo',
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logging.get('mode', 'disabled'),
    )
    for k, v in cfg.logging.items():
        if k != 'mode':
            init_kwargs[k] = v

    # Resume existing run
    if os.path.exists(swanlab_id_file):
        with open(swanlab_id_file) as f:
            sid = f.read().strip()
        if sid:
            init_kwargs['id'] = sid
            init_kwargs['resume'] = 'must'

    swanlab.init(**init_kwargs)
    swanlab.log({
        'eval/success_rate': success_rate,
        'eval/success_count': total_success,
        'eval/total_episodes': total_episodes,
    }, step=epoch)

    # Upload video
    if best_video and os.path.exists(best_video):
        gif_path = best_video.replace('.mp4', '.gif')
        os.system(f'ffmpeg -y -loglevel error -i {best_video} -r 10 {gif_path}')
        if os.path.exists(gif_path):
            caption = f'epoch{epoch} ({\"success\" if total_success > 0 else \"fail\"})'
            swanlab.log({'eval/video': swanlab.Video(gif_path, caption=caption)}, step=epoch)
            os.remove(gif_path)

    swanlab.finish()
except Exception as e:
    print(f'\033[91m[Eval] SwanLab logging failed: {e}\033[0m')
"

    # Print result
    local result_json="${OUTPUT_DIR}/eval_epoch${epoch_num}.json"
    if [[ -f "${result_json}" ]]; then
        local succ=$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['success_count'])")
        local total=$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['total_episodes'])")
        local rate=$(python3 -c "import json; d=json.load(open('${result_json}')); print(f\"{d['success_rate']*100:.1f}\")")
        log "Eval result (epoch ${epoch_num}): ${succ}/${total} = ${rate}%"
    fi
}

# ======================== main loop ========================
log "============================================================"
log "Train-Eval Loop"
log "  cfg           : ${CFG_FILE}"
log "  hydra_config  : ${HYDRA_CONFIG_NAME}"
log "  task          : ${TASK_NAME} / ${TASK_CONFIG}"
log "  zarr          : ${ZARR_PATH}"
log "  epochs        : ${TOTAL_EPOCHS}"
log "  eval_every    : ${EPOCHS_PER_ROUND}"
log "  device        : ${GPU_IDS} (${NUM_TRAIN_GPUS} GPUs)"
log "  eval_episodes : ${TOTAL_EVAL_EPISODES} total"
log "  output_dir    : ${OUTPUT_DIR}"
log "============================================================"

DONE_EPOCHS=$(current_epoch)
log "Starting from epoch ${DONE_EPOCHS}"

while [[ ${DONE_EPOCHS} -lt ${TOTAL_EPOCHS} ]]; do
    REMAINING=$(( TOTAL_EPOCHS - DONE_EPOCHS ))
    THIS_ROUND=$(( EPOCHS_PER_ROUND < REMAINING ? EPOCHS_PER_ROUND : REMAINING ))

    log "---------- Round: train ${THIS_ROUND} epochs (${DONE_EPOCHS} -> $(( DONE_EPOCHS + THIS_ROUND ))) ----------"

    # ---- Train ----
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    cd "${SCRIPT_DIR}"

    # Resume SwanLab run if we have a saved run id
    SWANLAB_ARGS=""
    if [[ -f "${SWANLAB_ID_FILE}" ]]; then
        SWANLAB_ID=$(cat "${SWANLAB_ID_FILE}")
        if [[ -n "${SWANLAB_ID}" ]]; then
            SWANLAB_ARGS="logging.id=${SWANLAB_ID}"
            log "Resuming SwanLab run: ${SWANLAB_ID}"
        fi
    fi

    # Build Hydra overrides from train_cfg.yaml values
    HYDRA_OVERRIDES=(
        head_camera_type="${HEAD_CAMERA_TYPE}"
        training.dist_mode="${_cfg_training_dist_mode}"
        training.resume="${_cfg_training_resume}"
        training.num_epochs="${THIS_ROUND}"
        +training.lr_total_epochs="${_cfg_lr_total_epochs}"
        training.checkpoint_every="${THIS_ROUND}"
        task.dataset.zarr_path="${ZARR_PATH}"
        hydra.run.dir="${OUTPUT_DIR}"
        hydra.job.name="train_eval_${TASK_NAME}"

        # Training
        training.seed="${_cfg_training_seed}"
        training.lr_scheduler="${_cfg_training_lr_scheduler}"
        training.lr_warmup_steps="${_cfg_training_lr_warmup_steps}"
        training.gradient_accumulate_every="${_cfg_training_gradient_accumulate_every}"
        training.use_ema="${_cfg_training_use_ema}"
        training.freeze_encoder="${_cfg_training_freeze_encoder}"
        training.eval_every="${_cfg_training_eval_every}"
        training.eval_episodes="${_cfg_training_eval_episodes}"
        training.val_every="${_cfg_training_val_every}"
        training.sample_every="${_cfg_training_sample_every}"
        training.log_every="${_cfg_training_log_every}"
        training.debug="${_cfg_training_debug}"

        # Optimizer
        optimizer.lr="${_cfg_optimizer_lr}"
        "optimizer.betas=${_cfg_optimizer_betas}"
        optimizer.eps="${_cfg_optimizer_eps}"
        optimizer.weight_decay="${_cfg_optimizer_weight_decay}"

        # Dataloader
        dataloader.batch_size="${_cfg_dataloader_batch_size}"
        dataloader.num_workers="${_cfg_dataloader_num_workers}"
        dataloader.shuffle="${_cfg_dataloader_shuffle}"
        dataloader.pin_memory="${_cfg_dataloader_pin_memory}"
        dataloader.persistent_workers="${_cfg_dataloader_persistent_workers}"

        val_dataloader.batch_size="${_cfg_val_dataloader_batch_size}"
        val_dataloader.num_workers="${_cfg_val_dataloader_num_workers}"
        val_dataloader.shuffle="${_cfg_val_dataloader_shuffle}"
        val_dataloader.pin_memory="${_cfg_val_dataloader_pin_memory}"
        val_dataloader.persistent_workers="${_cfg_val_dataloader_persistent_workers}"

        # Policy
        horizon="${_cfg_policy_horizon}"
        n_obs_steps="${_cfg_policy_n_obs_steps}"
        n_action_steps="${_cfg_policy_n_action_steps}"
        n_latency_steps="${_cfg_policy_n_latency_steps}"
        obs_as_global_cond="${_cfg_policy_obs_as_global_cond}"
        policy.num_inference_steps="${_cfg_policy_num_inference_steps}"
        policy.diffusion_step_embed_dim="${_cfg_policy_diffusion_step_embed_dim}"
        "policy.down_dims=${_cfg_policy_down_dims}"
        policy.kernel_size="${_cfg_policy_kernel_size}"
        policy.n_groups="${_cfg_policy_n_groups}"
        policy.cond_predict_scale="${_cfg_policy_cond_predict_scale}"

        # Dataset
        task.dataset.val_ratio="${_cfg_dataset_val_ratio}"
        task.dataset.seed="${_cfg_training_seed}"

        # Logging
        logging.project="${_cfg_logging_project}"
        logging.mode="${_cfg_logging_mode}"
        logging.resume="${_cfg_logging_resume}"
        logging.name="${EXPERIMENT_NAME}"

        ${SWANLAB_ARGS}
    )

    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NUM_TRAIN_GPUS}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        train.py \
        --config-name="${HYDRA_CONFIG_NAME}" \
        "${HYDRA_OVERRIDES[@]}"

    TRAIN_EXIT=$?
    if [[ ${TRAIN_EXIT} -ne 0 ]]; then
        err "Training exited with code ${TRAIN_EXIT}. Aborting loop."
        exit ${TRAIN_EXIT}
    fi

    # ---- Check checkpoint ----
    LATEST_CKPT=$(find_latest_ckpt)
    if [[ -z "${LATEST_CKPT}" ]]; then
        warn "No checkpoint found after training round. Skipping eval."
    else
        DONE_EPOCHS=$(current_epoch)
        log "Checkpoint: ${LATEST_CKPT} (epoch ${DONE_EPOCHS})"

        # ---- Eval ----
        run_eval "${LATEST_CKPT}" "${DONE_EPOCHS}"
    fi

    DONE_EPOCHS=$(current_epoch)
    log "Progress: ${DONE_EPOCHS} / ${TOTAL_EPOCHS} epochs done"
done

log "============================================================"
log "All done! ${TOTAL_EPOCHS} epochs completed."
log "Output directory: ${OUTPUT_DIR}"
log "============================================================"
