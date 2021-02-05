import rospy
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection, RobotConnection
from .controllers_connection import ControllersConnection
from ur_openai.msg import RLExperimentInfo
from std_msgs.msg import Bool
import ur_control.log as utils
color_log = utils.TextColors()

# https://github.com/openai/gym/blob/master/gym/core.py


class RobotGazeboEnv(gym.Env):
    def __init__(self,
                 robot_name_space,
                 controllers_list,
                 reset_controls,
                 use_gazebo=False,
                 **kwargs):

        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        if use_gazebo:
            self.robot_connection = GazeboConnection(**kwargs)
        else:
            self.robot_connection = RobotConnection()
            self.subs_pause = rospy.Subscriber('ur3/pause', Bool, self.pause_callback, queue_size=1)

        self.controllers_object = ControllersConnection(
            namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.step_count = 0
        self.cumulated_episode_reward = 0
        self.pause = False
        self.reward_pub = rospy.Publisher('/openai/reward',
                                          RLExperimentInfo,
                                          queue_size=1)
        self._log_message = None
        rospy.logdebug("END init RobotGazeboEnv")

    def pause_callback(self, msg):
        if msg.data is True:
            self.pause = True

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """
        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        if self.pause:
            self._pause_env()
            self.pause = False

        self.robot_connection.unpause()
        self._set_action(action)
        self.robot_connection.pause()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        self.step_count += 1

        return obs, reward, done, info

    def reset(self):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._update_episode()
        self._init_env_variables()
        self._reset_sim()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        if self._log_message is not None:
            color_log.ok("\n>> End of Episode = %s, Reward= %s, steps=%s" %
                         (self.episode_num, self.cumulated_episode_reward, self.step_count))
            color_log.warning(self._log_message)

        rospy.logdebug("PUBLISHING REWARD...")
        self._publish_reward_topic(self.cumulated_episode_reward,
                                   self.episode_num)
        rospy.logdebug("PUBLISHING REWARD...DONE=" +
                       str(self.cumulated_episode_reward) + ",EP=" +
                       str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls:
            rospy.logdebug("RESET CONTROLLERS")
            self.robot_connection.unpause()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.robot_connection.pause()
            self.robot_connection.reset()
            self.robot_connection.unpause()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.robot_connection.pause()

        else:
            rospy.logdebug("DONT RESET CONTROLLERS")
            self.robot_connection.unpause()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.robot_connection.pause()
            self.robot_connection.reset()
            self.robot_connection.unpause()
            self._check_all_systems_ready()
            self.robot_connection.pause()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

    def _pause_env(self):
        """Perform any validation/checks before/after pausing environment"""
        raise NotImplementedError()
