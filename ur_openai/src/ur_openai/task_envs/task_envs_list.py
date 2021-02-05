#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs


def register_environment(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # Task-Robot Envs

    result = True

    if task_env == 'UR3eTaskSpaceEnv-v0':

        register(
            id=task_env,
            entry_point='ur_openai.task_envs.ur3e_task_space:UR3eTaskSpaceEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from ur_openai.task_envs import ur3e_task_space

    elif task_env == 'UR3eJointSpaceEnv-v0':

        register(
            id=task_env,
            entry_point='ur_openai.task_envs.ur3e_joint_space:UR3eJointSpaceEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from ur_openai.task_envs import ur3e_joint_space

    # Add here your Task Envs to be registered
    else:
        result = False
    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = get_all_registered_envs()
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result


def get_all_registered_envs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids
