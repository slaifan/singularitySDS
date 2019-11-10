"""
This module contains code to train an agent using the RLLib framework.

The generated model will be written to ./model in the TF saved_model format.

The ./policy.py will then read the model from ./model for evaluation.
"""

import os
import random
import argparse
import math
import shutil

import numpy as np
import gym

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

from hiway.sumo_scenario import SumoScenario

from gym_hiway.env.competition_env import CompetitionEnv
from hiway.utils import abs_path

from policy import MODEL_NAME, OBSERVATION_SPACE, ACTION_SPACE, observation, reward, action


# Add custom metrics to your tensorboard using these callbacks
# see: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["ego_speed"] = []


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)
    episode.user_data["ego_speed"].append(obs['speed'])


def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])
    print("episode {} ended with length {} and mean ego speed {:.2f}".format(
        episode.episode_id, episode.length, mean_ego_speed))
    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


def main(args):
    sumo_scenario = SumoScenario(
        scenario_root=os.path.abspath(args.scenario),
        random_social_vehicle_count=args.num_social_vehicles)

    tune_config = {
        'env': CompetitionEnv,
        'log_level': 'WARN',
        'num_workers': 2,
        'horizon': 5000,
        'env_config': {
            'seed': tune.randint(1000),
            'sumo_scenario': sumo_scenario,
            'headless': args.headless,
            'observation_space': OBSERVATION_SPACE,
            'action_space': ACTION_SPACE,
            'reward_function': tune.function(reward),
            'observation_function': tune.function(observation),
            'action_function': tune.function(action),
        },
        'model':  {
            'custom_model': MODEL_NAME,
        },
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end
        }
    }

    experiment_name = 'rllib_example'

    log_dir = os.path.expanduser("~/ray_results")
    print(f"Checkpointing at {log_dir}")
    analysis = tune.run(
        'PPO',
        name=experiment_name,
        stop={'time_total_s': 60 * 60}, # 1 hour
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=log_dir,
        resume=args.resume_training,
        max_failures=10,
        num_samples=args.num_samples,
        export_formats=['model', 'checkpoint'],
        config=tune_config,)

    print(analysis.dataframe().head())

    logdir = analysis.get_best_logdir('episode_reward_max')
    model_path = os.path.join(logdir, 'model')
    dest_model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "model")

    if not os.path.exists(dest_model_path):
        shutil.copytree(model_path, dest_model_path)
        print(f"wrote model to: {dest_model_path}")
    else:
        print(f"Model already exists at {dest_model_path} not overwriting")
        print(f"New model is stored at {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rllib-example')
    parser.add_argument('scenario',
                        help='Scenario to run (see scenarios/ for some samples you can use)',
                        type=str)
    parser.add_argument('--headless',
                        default=False,
                        help='run simulation in headless mode',
                        action='store_true')
    parser.add_argument('--num_samples',
                        type=int,
                        default=1,
                        help='Number of times to sample from hyperparameter space')
    parser.add_argument('--num_social_vehicles',
                        type=int,
                        default=0,
                        help='Number of social vehicles in the environment')
    parser.add_argument('--resume_training',
                        default=False,
                        action='store_true',
                        help='Resume the last trained example')
    args = parser.parse_args()
    main(args)
