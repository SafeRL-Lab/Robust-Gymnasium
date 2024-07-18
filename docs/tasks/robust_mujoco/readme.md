$$S = S + \epsilon_g \tag{1.1} $$

$$
\epsilon_g \in N(\mu, \sigma) \tag{1.2}
$$

$$S = S + \epsilon_u \tag{1.3} $$

$$
u \sim \mathcal{U}(a, b) \tag{1.4}
$$

$$S = S + \epsilon_{sh} \tag{1.5} $$

$$
\epsilon_{sh} = c  \tag{1.6} 
$$

## Detailed Environments and Tasks


In the equation $(1.1)$

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
| Robust Ant Tasks|   In the task, the robot observation, reward signal, action, joint force and mass (shape) could be randomly changed with a Gaussian distribution, uniform distribution, or a certain shift.| <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Ant-v4.gif" width="250"/>    |
| Robust HalhCheetah Tasks| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/HalfCheetah-v4.gif" align="middle" width="250"/>    |
|Robust Hopper Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Hopper-v4.gif" align="middle" width="250"/>    |
|Robust Pusher Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Pusher-v4.gif" align="middle" width="250"/>    |
|Robust Reacher Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Reacher-v4.gif" align="middle" width="250"/>    |
|Robust Swimmer Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Swimmer-v4.gif" align="middle" width="250"/>    |
|Robust Walker Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Walker2d-v4.gif" align="middle" width="250"/>    |
|Robust Humanoid Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoid.gif" align="middle" width="250"/>    |
|Robust HumanoidStandup Tasks| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoidStandup-v4.gif" align="middle" width="250"/>    |
|Robust Invert-Double Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/inver-double.gif" align="middle" width="250"/>    |
|Robust Invert Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/invert.gif" align="middle" width="250"/>    |




<details>
<summary><b><big>Details of Robust MuJoCo Tasks</big></b></summary>

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
| Robust Ant Tasks|   In the task, the robot observation, reward signal, action, joint force and mass (shape) could be randomly changed with a Gaussian distribution, uniform distribution, or a certain shift.| <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Ant-v4.gif" width="250"/>    |
| Robust HalhCheetah Tasks| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/HalfCheetah-v4.gif" align="middle" width="250"/>    |
|Robust Hopper Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Hopper-v4.gif" align="middle" width="250"/>    |
|Robust Pusher Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Pusher-v4.gif" align="middle" width="250"/>    |
|Robust Reacher Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Reacher-v4.gif" align="middle" width="250"/>    |
|Robust Swimmer Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Swimmer-v4.gif" align="middle" width="250"/>    |
|Robust Walker Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Walker2d-v4.gif" align="middle" width="250"/>    |
|Robust Humanoid Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoid.gif" align="middle" width="250"/>    |
|Robust HumanoidStandup Tasks| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoidStandup-v4.gif" align="middle" width="250"/>    |
|Robust Invert-Double Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/inver-double.gif" align="middle" width="250"/>    |
|Robust Invert Tasks |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/invert.gif" align="middle" width="250"/>    |
</details>


<details>
<summary><b><big>Details of Robust Robot Manipulation Tasks</big></b></summary>

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
|Multi-Robot Door Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/multi-robot-door-small.gif" width="250"/>    |
|Nominal Door Tasks|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Nominal_task%2000_00_00-00_00_30.gif" width="250"/>    |
| Shifted Door Task 1| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_1%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 2|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_2%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 3|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_3%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 4|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_4%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 5 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_5%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 6 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_6%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 7 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_7%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 8| | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_8%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 9 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_9%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 10 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_10%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 11 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_11%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 12 |  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_12%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
|Shifted Door Task 13|  | <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_13%2000_00_00-00_00_30.gif" align="middle" width="250"/>    |
</details>

<details>
<summary><b><big>Details of Robust Dexterous Hand and Maze Tasks</big></b></summary>
  
TBD
</details>

<details>
<summary><b><big>Details of Robust Safety Tasks</big></b></summary>
  
TBD
</details>

<details>
<summary><b><big>Details of Robust MuJoCo Variant Tasks</big></b></summary>
  
TBD
</details>


--------------
