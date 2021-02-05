from ur_openai import robot_env
import rospy
import numpy as np
from numpy.random import RandomState

from ur_control.arm import Arm

class UREnv(robot_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """

        rospy.logdebug("Start UREnv Init")
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['arm_controller', 'gripper_controller']

        # It doesnt use namespace
        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_env.RobotGazeboEnv

        super(UREnv, self).__init__(controllers_list=self.controllers_list,
                                        robot_name_space=self.robot_name_space,
                                        reset_controls=reset_controls_bool,
                                        use_gazebo=self.param_use_gazebo,
                                        start_init_physics_parameters=False,
                                        reset_world_or_sim="WORLD")
        self.robot_connection.unpause()

        rospy.logdebug("UREnv unpause...")

        self.ur3e_arm = Arm(ft_sensor=self.ft_sensor,
                            driver=self.driver,
                            ee_transform=self.extra_ee.tolist())

        if self.rand_seed is not None:
            self.seed(self.rand_seed)
            RandomState(self.rand_seed)
            np.random.seed(self.rand_seed)

        rospy.logdebug("Finished UREnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _pause_env(self):
        current_pose = self.ur_arm.joint_angles()
        input("Press Enter to continue")
        self.ur_arm.set_joint_positions(current_pose, wait=True, t=self.reset_time)


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
