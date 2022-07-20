import tensorflow as tf
import functools
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import reverb
print('current directory', os.getcwd())

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.trajectories import TimeStep
from tf_agents.environments import suite_gym
from circuit_training.environment import environment
from circuit_training.environment import plc_client
from circuit_training.model import model
from circuit_training.learning import agent

'''
    RUN COMMAND: python3 -m circuit_training.policy_reload
'''

#flags.DEFINE_string('plc_wrapper_main', 'plc_wrapper_main','Path to plc_wrapper_main binary.')
print("*** Reloading policy ****")
# non-fatal warning https://github.com/tensorflow/tensorflow/issues/42738
policy_path = '/home/yuw/Desktop/Github/CT_runnable/train_policies/policy/'
# policy_path = '../logs/run_00/111/policies/collect_policy'

# testcase path
_NETLIST_FILE = flags.DEFINE_string('netlist_file', "./circuit_training/environment/test_data/ariane/netlist.pb.txt",
                                    'File path to the netlist file.')
_INIT_PLACEMENT = flags.DEFINE_string('init_placement', "./circuit_training/environment/test_data/ariane/initial.plc",
                                      'File path to the init placement file.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', "./test_placement.plc", "File path to the output placement file.")
_GLOBAL_SEED = flags.DEFINE_integer(
    'global_seed', 111,
    'Used in env and weight initialization, does not impact action sampling.')

FLAGS = flags.FLAGS

def main(_argv):
    logging.info('global seed=%d', _GLOBAL_SEED.value)
    logging.info('netlist_file=%s', _NETLIST_FILE.value)
    logging.info('init_placement=%s', _INIT_PLACEMENT.value)

    # initialize test environment
    create_env_fn = functools.partial(
        environment.create_circuit_environment,
        netlist_file=_NETLIST_FILE.value,
        init_placement=_INIT_PLACEMENT.value,
        global_seed=_GLOBAL_SEED.value)

    test_env = create_env_fn()

    print(test_env.time_step_spec())
    print(test_env.action_spec())

    # env = environment.CircuitEnv(
    #     netlist_file=_NETLIST_FILE.value,
    #     init_placement=_INIT_PLACEMENT.value,
    #     is_eval=True,
    #     save_best_cost=True,
    #     output_plc_file=_OUTPUT_PATH.value,
    #     cd_finetune=True,
    #     train_step=train_step)

    # reload policy checkpoint
    policy_checkpointer = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_path, test_env.time_step_spec(), test_env.action_spec(), load_specs_from_pbtxt=True)
    print("policy reloaded:", policy_checkpointer)

    # [Optional] get all attibutes from policy checkpoint
    META_INFO = False

    if META_INFO:
        print(policy_checkpointer.__dict__)
        print(dir(policy_checkpointer))

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(test_env))
    static_features = test_env.get_static_obs()

    strategy = tf.distribute.get_strategy()

    # print("static_feature", static_features)
    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)

    # Create the agent whose collect policy is being tested.
    with strategy.scope():
        train_step = train_utils.create_train_step()
        tf_agent = agent.create_circuit_ppo_grl_agent(train_step,
                                                    action_tensor_spec,
                                                    time_step_tensor_spec,
                                                    grl_actor_net,
                                                    grl_value_net,
                                                    strategy)
    tf_agent.initialize()

    # inference test
    print("*** Inference Testing ****")

    # [Optional] Consruct time step manually: https://stackoverflow.com/questions/57565249/valueerror-could-not-find-matching-function-to-call-loaded-from-the-savedmodel
    # time_step = test_env.reset()
    
    observation = test_env.reset()

    step_type = tf.convert_to_tensor(
        [0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [1], dtype=tf.float32, name='discount')
    time_step = TimeStep(step_type, reward, discount, observation)
    
    action_res = policy_checkpointer.action(time_step=observation)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass