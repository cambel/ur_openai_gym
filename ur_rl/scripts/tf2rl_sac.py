#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #tensorflow logging disabled
from shutil import copyfile
import rospy
import timeit
import argparse
from gym.envs.registration import register

from tf2rl.algos.sac import SAC
from tf2rl.experiments.cb_trainer import Trainer

import ur_control.utils as utils

from ur_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params
import numpy as np
np.set_printoptions(suppress=True)

import sys
import signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser.add_argument('-e', '--env_id', type=int, help='environment ID', default=None)
    parser.set_defaults(batch_size=2048)
    parser.set_defaults(n_warmup=0) # still don't know what it this for
    parser.set_defaults(max_steps=10000) # 10000 for training 200 for evaluation
    parser.set_defaults(save_model_interval=10000)
    parser.set_defaults(test_interval=1e10) # 1e4 for training 200 for evaluation
    parser.set_defaults(test_episodes=1)
    parser.set_defaults(normalize_obs=False)
    parser.set_defaults(auto_alpha=False)
    parser.set_defaults(use_prioritized_rb=True)
    parser.set_defaults(lr=3e-4)

    args = parser.parse_args(rospy.myargv()[1:])


    rospy.init_node('ur3e_tf2rl',
                    anonymous=True,
                    log_level=rospy.ERROR)

    clear_gym_params('ur_gym')

    start_time = timeit.default_timer()

    args = parser.parse_args()
    param_file = None
    
    if args.evaluate:
        args.n_warmup = 0
        args.max_steps = 150
        args.test_interval = 1
        args.test_episodes = 10

    if args.env_id == 0:
        args.dir_suffix = "task_space"
        param_file = "simulation/task_space.yaml"
    elif args.env_id == 1:
        args.dir_suffix = "joint_space"
        param_file = "simulation/joint_space.yaml"
    else:
        raise Exception("invalid env_id")

    p = utils.TextColors()
    p.error("GYM Environment:{} ".format(param_file))

    ros_param_path = load_ros_params(rospackage_name="ur_rl",
                    rel_path_from_package_to_file="config",
                    yaml_file_name=param_file)

    args.episode_max_steps = rospy.get_param("ur_gym/steps_per_episode", 200)

    env = load_environment(
            rospy.get_param('ur_gym/env_id'),
            max_episode_steps=args.episode_max_steps)
    actor_class = rospy.get_param("ur_gym/actor_class", "default")

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        actor_class=actor_class,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        auto_alpha=args.auto_alpha,
        lr=args.lr
        )
    trainer = Trainer(policy, env, args, test_env=None)
    outdir = trainer._output_dir
    rospy.set_param('ur_gym/output_dir', outdir)
    log_ros_params(outdir)
    copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")
    trainer()

    print("duration", (timeit.default_timer() - start_time)/60.,"min")

# rosrun ur3e_rl tf2rl_sac.py --env-id=0