import os, sys

import gym
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

from ray.rllib.utils import try_import_tf
tf = try_import_tf()


MODEL_NAME = 'model'


# This action space should match the input to the action(..) function below.
ACTION_SPACE = gym.spaces.Box(low=np.array([0.0, 0.0, -1.0]),
                              high=np.array([1.0, 1.0, 1.0]),
                              dtype=np.float32)


# The maximum number of lanes we expect to see in any scenario.
MAX_LANES = 5


# This observation space should match the output of observation(..) below
OBSERVATION_SPACE = gym.spaces.Dict({
    'distance_from_center': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'angle_error': gym.spaces.Box(low=-180, high=180, shape=(1,)),
    'speed': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'steering': gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
    'lane_idx': gym.spaces.Discrete(MAX_LANES),
})


def observation(env_obs):
    """
    Transform the environment's observation into something more suited for your model
    """
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return {
        'distance_from_center': np.array([norm_dist_from_center]),
        'angle_error': np.array([closest_wp.relative_heading(ego.heading)]),
        'speed': np.array([ego.speed]),
        'steering': np.array([ego.steering]),
        'lane_idx': closest_wp.lane_index,
    }


def reward(env_obs, env_reward):
    """
    Here you can perform your reward shaping.

    The default reward provided by the environment is the increment in
    distance travelled. Your model will likely require a more
    sophisticated reward function
    """
    return env_reward

def action(model_action):
    """
    Here you can perform you action space shaping.

    The built in action space may not be suited for all approaches, for
    example you may like to use a discrete action space.
    """
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering * 45])


# See: https://github.com/ray-project/ray/blob/b89cac976ae171d6d9b3245394e4932288fc6f11/rllib/models/tf/fcnet_v2.py#L14
class Model(FullyConnectedNetwork):
    pass


ModelCatalog.register_custom_model(MODEL_NAME, Model)


class EvaluationModel(TFModelV2):
    def __init__(self):
        super().__init__(
            obs_space=OBSERVATION_SPACE,
            action_space=ACTION_SPACE,
            num_outputs=ACTION_SPACE.shape[0],
            model_config={},
            name="evaluation-model")

        self._prep = ModelCatalog \
            .get_preprocessor_for_space(OBSERVATION_SPACE)

        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        model_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'model')

        self.base_model = tf.saved_model.load(self._sess,
                                              export_dir=model_path,
                                              tags=['serve'])

    def forward(self, obs, state, seq_lens):
        obs = self._prep.transform(obs)
        graph = tf.get_default_graph()
        # These tensor names were found by inspecting the trained model
        output_node = graph.get_tensor_by_name("default_policy/add:0")
        input_node = graph.get_tensor_by_name("default_policy/observation:0")
        res = self._sess.run(output_node, feed_dict={input_node: [obs]})
        action = res[0]
        return action, state


class Policy():
    def setup(self):
        self._model = EvaluationModel()

    def teardown(self):
        pass

    def act(self, observation):
        action, _state = self._model.forward(observation, None, None)
        return action
