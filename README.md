
## Demos


<p align="center">
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Ant-v4.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/HalfCheetah-v4.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Hopper-v4.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>
   
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Pusher-v4.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Reacher-v4.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Swimmer-v4.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>

  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/Walker2d-v4.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoid.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/humanoidStandup-v4.gif" alt="Image 3" width="300" style="margin-right: 10px;"/> 
  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/inver-double.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/invert.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
</p>
<p align="center">
  These demonstrations are from version 4 of the MuJoCo tasks with robust settings.
</p>

---------------

We are currently developing our environments on an Ubuntu system. The operating system version is 20.04.3 LTS.

## Setting Up the Environment

1. **Create an environment (requires [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):**

   Use the following command to create a new Conda environment named `robustgymnasium` with Python 3.10:

   ```bash
   conda create -n robustgymnasium  python=3.10
   ```

   Activate the newly created environment:

   ```bash
   conda activate robustgymnasium
   ```

2. **Install dependency packages:**

   Install the necessary packages using pip. Make sure you are in the project directory where the `setup.py` file is located:

   ```bash
   pip install gymnasium[mujoco]
   pip install -e .
   ```

## Testing the Tasks

To run the tests, navigate to the `examples` directory and execute the test script:

```bash
cd examples/
python test.py
```

Ensure you follow these steps to set up and test the environment properly. Adjust paths and versions as necessary based on your specific setup requirements.

---




## Tasks
- [x] Single Agent Tasks
- [ ] Multi-Agent Tasks

### Single Robust Agent Tasks

| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| Ant-v4         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| HalfCheetah-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Hopper-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Walker2d-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Swimmer-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Humanoid-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| HumanoidStandup-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Pusher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Reacher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedDoublePendulum-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedPendulum-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
|  |   |  |  |
| Ant-v2         |    :white_check_mark:     |    :white_check_mark:      |:white_check_mark:          |
| HalfCheetah-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Hopper-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Walker2d-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Swimmer-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Humanoid-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| HumanoidStandup-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Pusher-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Reacher-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedDoublePendulum-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedPendulum-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
|  |   |  |  |
| Ant-v3         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| HalfCheetah-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Hopper-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Walker2d-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Swimmer-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Humanoid-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| HumanoidStandup-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Pusher-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Reacher-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedDoublePendulum-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedPendulum-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
|  |   |  |  |
| Ant-v5         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| HalfCheetah-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Hopper-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Walker2d-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Swimmer-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Humanoid-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| HumanoidStandup-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Pusher-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Reacher-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedDoublePendulum-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| InvertedPendulum-v5    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |


### Safety Robust Rasks
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| SafetyAnt-v4         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| SafetyAntHalfCheetah-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntHopper-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntWalker2d-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntSwimmer-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntHumanoid-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntHumanoidStandup-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntPusher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| SafetyAntReacher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

---------







## Acknowledgments

We thank the contributors from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium.git).



