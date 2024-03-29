import datetime
import rospy
import numpy as np

from gym import spaces

from ur_control import transformations, spalg

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur_env
from ur_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose, apply_workspace_contraints


class UR3eTaskSpaceEnv(ur_env.UREnv):
    def __init__(self):

        self.cost_positive = False
        self.get_robot_params()

        ur_env.UREnv.__init__(self)

        self._previous_joints = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None

        self.last_actions = np.zeros(self.n_actions)
        obs = self._get_obs()

        self.reward_threshold = 500.0

        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions, ),
                                       dtype='float32')

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs.shape,
                                            dtype='float32')

        self.trials = 1
        self.rate = rospy.Rate(1/self.agent_control_dt)

        print("ACTION SPACES TYPE", (self.action_space))
        print("OBSERVATION SPACES TYPE", (self.observation_space))

    def get_robot_params(self):
        prefix = "ur_gym"
        load_param_vars(self, prefix)

        driver_param = rospy.get_param(prefix + "/driver")
        self.param_use_gazebo = False
        if driver_param == "robot":
            self.param_use_gazebo = False

        self.relative_to_ee = rospy.get_param(prefix + "/relative_to_ee", False)

        self.target_pose_uncertain = rospy.get_param(prefix + "/target_pose_uncertain", False)
        self.fixed_uncertainty_error = rospy.get_param(prefix + "/fixed_uncertainty_error", False)
        self.target_pose_uncertain_per_step = rospy.get_param(prefix + "/target_pose_uncertain_per_step", False)
        self.true_target_pose = rospy.get_param(prefix + "/target_pos", False)
        self.rand_seed = rospy.get_param(prefix + "/rand_seed", None)
        self.rand_init_interval = rospy.get_param(prefix + "/rand_init_interval", 5)
        self.rand_init_counter = 0
        self.rand_init_cpose = None

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        :return: observations
        """
        joint_angles = self.ur3e_arm.joint_angles()
        ee_points, ee_velocities = self.get_points_and_vels(joint_angles)

        obs = np.concatenate([
            ee_points.ravel(),  # [6]
            ee_velocities.ravel(),  # [6]
        ])

        return obs.copy()

    def get_points_and_vels(self, joint_angles):
        """
        and velocities of the end effector with respect to the target pose
        """

        if self._previous_joints is None:
            self._previous_joints = self.ur3e_arm.joint_angles()

        # Current position
        ee_pos_now = self.ur3e_arm.end_effector(joint_angles=joint_angles)

        # Last position
        ee_pos_last = self.ur3e_arm.end_effector(joint_angles=self._previous_joints)
        self._previous_joints = joint_angles  # update

        # Use the past position to get the present velocity.
        linear_velocity = (ee_pos_now[:3] - ee_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            ee_pos_now[3:], ee_pos_last[3:], self.agent_control_dt)
        velocity = np.concatenate((linear_velocity, angular_velocity))

        # Shift the present position by the End Effector target.
        # Since we subtract the target point from the current position, the optimal
        # value for this will be 0.
        error = spalg.translation_rotation_error(self.target_pos, ee_pos_now)

        return error, velocity

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self._log()
        cpose = self.ur3e_arm.end_effector()
        deltax = np.array([0., 0., 0.02, 0., 0., 0.])
        cpose = transformations.pose_euler_to_quaternion(
            self.ur3e_arm.end_effector(), deltax, ee_rotation=True)
        self.ur3e_arm.set_target_pose(pose=cpose,
                                      wait=True,
                                      t=self.reset_time)
        self._add_uncertainty_error()
        if self.random_initial_pose:
            self._randomize_initial_pose()
            self.ur3e_arm.set_target_pose(pose=self.rand_init_cpose,
                                          wait=True,
                                          t=self.reset_time)
        else:
            qc = self.init_q
            self.ur3e_arm.set_joint_positions(position=qc,
                                              wait=True,
                                              t=self.reset_time)
        self.max_distance = spalg.translation_rotation_error(self.ur3e_arm.end_effector(), self.target_pos) * 1000.
        self.max_dist = None

    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(self.ur3e_arm.end_effector(self.init_q), self.workspace, self.reset_time)
            self.rand_init_counter = 0
        self.rand_init_counter += 1

    def _add_uncertainty_error(self):
        if self.target_pose_uncertain:
            if len(self.uncertainty_std) == 2:
                translation_error = np.random.normal(scale=self.uncertainty_std[0], size=3)
                translation_error[2] = 0.0
                rotation_error = np.random.normal(scale=self.uncertainty_std[1], size=3)
                rotation_error = np.deg2rad(rotation_error)
                error = np.concatenate([translation_error, rotation_error])
            elif len(self.uncertainty_std) == 6:
                if self.fixed_uncertainty_error:
                    error = self.uncertainty_std.copy()
                    error[3:] = np.deg2rad(error[3:])
                else:
                    translation_error = np.random.normal(scale=self.uncertainty_std[:3])
                    rotation_error = np.random.normal(scale=self.uncertainty_std[3:])
                    rotation_error = np.deg2rad(rotation_error)
                    error = np.concatenate([translation_error, rotation_error])
            else:
                print("Warning: invalid uncertanty error", self.uncertainty_std)
                return
            self.target_pos = transformations.pose_euler_to_quaternion(self.true_target_pose, error, ee_rotation=True)

    def _log(self):
        # Test
        # log_data = np.array([self.controller.force_control_model.update_data,self.controller.force_control_model.error_data])
        # print("Hellooo",log_data.shape)
        # logfile = rospy.get_param("ur3e_gym/output_dir") + "/log_" + \
        #             datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
        # np.save(logfile, log_data)
        if self.obs_logfile is None:
            try:
                self.obs_logfile = rospy.get_param("ur3e_gym/output_dir") + "/state_" + \
                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
            except Exception:
                return
        # save_log(self.obs_logfile, self.obs_per_step, self.reward_per_step, self.cost_ws)
        self.reward_per_step = []
        self.obs_per_step = []

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        state = observations.ravel()
        self.obs_per_step.append([state])

        if self.reward_type == 'sparse':
            return cost.sparse(self, done)
        elif self.reward_type == 'distance':
            return cost.distance(self, observations, "l1l2")
        else:
            raise AssertionError("Unknown reward function", self.reward_type)

        return 0

    def _is_done(self, observations):
        if self.target_pose_uncertain_per_step:
            self._add_uncertainty_error()

        true_error = spalg.translation_rotation_error(self.true_target_pose, self.ur3e_arm.end_effector())
        true_error[:3] *= 1000.0
        true_error[3:] = np.rad2deg(true_error[3:])
        success = np.linalg.norm(true_error[:3], axis=-1) < self.distance_threshold
        self._log_message = "Final distance: " + str(np.round(true_error, 3)) + (' success!' if success else '')
        return success

    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)

    def _set_action(self, action):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        # ensure that we don't change the action outside of this scope
        actions = np.copy(action)

        if self.n_actions == 3:

            cmd = self.ur3e_arm.end_effector()
            cmd[:3] += actions * 0.001

        if self.n_actions == 4:

            cpose = self.ur3e_arm.end_effector()

            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0], delta[3:])) # Do not change ax and ay

            # cmd = cpose + delta 
            cmd = transformations.pose_euler_to_quaternion(cpose, actions)
            cmd = apply_workspace_contraints(cmd, self.workspace)


        if self.n_actions == 6:

            cpose = self.ur3e_arm.end_effector()

            delta = actions * 0.001

            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
            # print("cmd", cmd)
            # cmd = apply_workspace_contraints(cmd, self.workspace)
            # print("cmd2", cmd)

        self.ur3e_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)
        self.rate.sleep()
