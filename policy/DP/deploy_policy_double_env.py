import numpy as np
try:
    from .dp_model import DP
except:
    pass

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    # Add depth observations (used by rgbd configs B and C, ignored by baseline)
    head_depth = observation["observation"]["head_camera"]["depth"].astype(np.float32)
    obs["head_depth"] = head_depth[np.newaxis]  # (1, H, W)
    if "left_camera" in observation["observation"]:
        left_depth = observation["observation"]["left_camera"]["depth"].astype(np.float32)
        right_depth = observation["observation"]["right_camera"]["depth"].astype(np.float32)
        obs["left_depth"] = left_depth[np.newaxis]
        obs["right_depth"] = right_depth[np.newaxis]
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    return DP(ckpt_file)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.call(func_name='get_action', obs=obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.call(func_name='update_obs', obs=obs)
