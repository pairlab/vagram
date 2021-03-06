# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from mbrl.models.vaml_mlp import VAMLMLP
import os
from typing import Optional, Tuple, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: pytorch_sac.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    obs = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(obs.shape[0], dtype=bool)
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(action, sample=True)
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_rewards[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_dones[~accum_dones],
            pred_dones[~accum_dones],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: pytorch_sac.Agent,
    num_episodes: int,
    video_recorder: pytorch_sac.VideoRecorder,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[pytorch_sac.ReplayBuffer],
    new_capacity: int,
    obs_shape: Tuple[int],
    act_shape: Tuple[int],
    device: torch.device,
) -> pytorch_sac.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        new_buffer = pytorch_sac.ReplayBuffer(
            obs_shape, act_shape, new_capacity, device
        )
        if sac_buffer is None:
            return new_buffer
        n = len(sac_buffer)
        new_buffer.add_batch(
            sac_buffer.obses[:n],
            sac_buffer.actions[:n],
            sac_buffer.rewards[:n],
            sac_buffer.next_obses[:n],
            torch.logical_not(sac_buffer.not_dones[:n]),
            torch.logical_not(sac_buffer.not_dones_no_max[:n]),
            from_np=False
        )
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = hydra.utils.instantiate(cfg.algorithm.agent)

    work_dir = work_dir or os.getcwd()

    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = pytorch_sac.VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)


    # insert annoying preemption code here

    # save all model and trainer chpts
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng, device=cfg.device
    )

    try:
        with open(os.path.join(work_dir, "epoch.txt"), "r") as f:
            env_steps = int(f.read())
        reloaded = True
    except FileNotFoundError as e:
        env_steps = 0
        reloaded = False
    
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    if reloaded:
        print("\n\nLoading from disk\n\n")
        replay_buffer.load(work_dir)
        dynamics_model.load(work_dir)
        agent.load(work_dir)
        optim_model_weights = torch.load(os.path.join(work_dir, "model_optim.pth"))
        model_trainer.optimizer.load_state_dict(optim_model_weights)
        sac_buffer = pytorch_sac.ReplayBuffer(obs_shape, act_shape, 0, torch.device(cfg.device))
        sac_buffer.load(work_dir)
    else:
        random_explore = cfg.algorithm.random_initial_explore
        mbrl.util.common.rollout_agent_trajectories(
            env,
            cfg.algorithm.initial_exploration_steps,
            mbrl.planning.RandomAgent(env) if random_explore else agent,
            {} if random_explore else {"sample": True, "batched": False},
            replay_buffer=replay_buffer,
        )
        replay_buffer.save(work_dir)
        dynamics_model.save(work_dir)
        agent.save(work_dir)
        torch.save(model_trainer.optimizer.state_dict(), os.path.join(work_dir, "model_optim.pth"))
        with open(os.path.join(work_dir, "epoch.txt"), "w") as f:
            f.write(str(env_steps))
        sac_buffer = None
    
    if isinstance(dynamics_model.model, VAMLMLP):
        dynamics_model.model.set_gradient_buffer(obs_shape, act_shape, cfg)
        dynamics_model.model.set_agent(agent)
        # add mse for first epoch
        dynamics_model.model.add_mse = True


    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = env_steps * cfg.overrides.num_sac_updates_per_step
    epoch = env_steps // cfg.overrides.epoch_length

    best_eval_reward = -np.inf

    while env_steps < cfg.overrides.num_steps:
        print(env_steps)
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer,
            sac_buffer_capacity,
            obs_shape,
            act_shape,
            torch.device(cfg.device),
        )
        obs, done = None, False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                if isinstance(dynamics_model.model, VAMLMLP):
                    print("Adding agent")
                    dynamics_model.model.set_gradient_buffer(obs_shape, act_shape, cfg)
                    dynamics_model.model.set_agent(agent)
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
                agent.save(work_dir)
                sac_buffer.save(work_dir)
                with open(os.path.join(work_dir, "epoch.txt"), "w") as f:
                    f.write(str(env_steps))

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    sac_buffer
                ) < rollout_batch_size:
                    break  # only update every once in a while
                agent.actor.requires_grad = True
                agent.critic.requires_grad = True
                model_data_likelihood = mbrl.util.math.truncated_linear(
                        *(cfg.overrides.model_data_likelihood + [epoch + 1])
                    )
                sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch

                buffer_choice = np.random.choice([False, True], p=[model_data_likelihood, 1.-model_data_likelihood])
                if buffer_choice:
                    agent.update(replay_buffer, logger, updates_made)
                else:
                    agent.update(sac_buffer, logger, updates_made)
                updates_made += 1

                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                video_recorder.save(f"{epoch}.mp4")
                best_eval_reward = avg_reward
                epoch += 1

                if isinstance(dynamics_model.model, VAMLMLP):
                    dynamics_model.model.add_mse = False

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
