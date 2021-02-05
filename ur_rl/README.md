# Tutorial for using RL with UR3e + force control on simulation


## Peg-in-hole example environment

<img src="https://drive.google.com/uc?export=view&id=1AzydhTNjup9C83oEjI0JmvaGGZCeKPnb" width="400">

1. Start the gazebo peg-in-hole environment 
    ```
    $ roslaunch ur3_gazebo ur3e_dual.launch peg_shape:=cube
    ```
2. Unpause the environment
3. Execute trainer script
   ```
   $ rosrun ur3e_rl tf2rl_sac.py --env-id 1
   ```
   This script will start a peg-in-hole training using domain randomization where the initial position of the robot, the position of the task board and the stiffness of the task board will change every few episodes. The configuration file for this training environment can be found in ur3e_rl/config/simulation/peg_in_hole.yaml
   The training session will last 5000 steps as define in the parameters of the tf2rl_sac.py script where each episode has at most 300 steps as defines in the peg_in_hole.yaml
   The trainer script support several parameters that can be seen with the command
   ```
   $ rosrun ur3e_rl tf2rl_sac.py --help
   ```

## Testing the learned policy
The learned policy will be safe in the folder (current folder)/results/time_env_name/*
The learned policy can be executed using the following command
```
$ rosrun ur3e_rl evaluate_policy.py results/time_env_name -n 2
```
the parameter -n is the number of trials to be executed. The policy is tested using the same parameters that were defined during training but that are saved to the file ros_gym_env_params.yaml located inside the policy folder.