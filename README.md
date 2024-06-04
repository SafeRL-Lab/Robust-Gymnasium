

We are currently developing our environments on an Ubuntu system. The operating system version is 20.04.3 LTS.

### Setting Up the Environment

1. **Create an environment (requires [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):**

   Use the following command to create a new Conda environment named `omnirobust` with Python 3.10:

   ```bash
   conda create -n omnirobust python=3.10
   ```

   Activate the newly created environment:

   ```bash
   conda activate omnirobust
   ```

2. **Install dependency packages:**

   Install the necessary packages using pip. Make sure you are in the project directory where the `setup.py` file is located:

   ```bash
   pip install -e .
   ```

### Testing the Tasks

To run the tests, navigate to the `tests` directory and execute the test script:

```bash
cd examples/
python test.py
```

Ensure you follow these steps to set up and test the environment properly. Adjust paths and versions as necessary based on your specific setup requirements.

---




### Tasks

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
| Ant-v2         |    :white_check_mark:     | :x:          |:x:          |
| HalfCheetah-v2    | :white_check_mark:      | :x:    | :x:          |
| Hopper-v2    | :white_check_mark:      | :x:    | :x:          |
| Walker2d-v2    | :white_check_mark:      | :x:    | :x:          |
| Swimmer-v2    | :white_check_mark:      | :x:    | :x:          |
| Humanoid-v2    | :white_check_mark:      | :x:    | :x:          |
| HumanoidStandup-v2    | :white_check_mark:      | :x:    | :x:          |
| Pusher-v2    | :white_check_mark:      | :x:    | :x:          |
| Reacher-v2    | :white_check_mark:      | :x:    | :x:          |
| InvertedDoublePendulum-v2    | :white_check_mark:      | :x:    | :x:          |
| InvertedPendulum-v2    | :white_check_mark:      | :x:    | :x:          |
|  |   |  |  |
| Ant-v3         |    :white_check_mark:     | :x:          |:x:          |
| HalfCheetah-v3    | :white_check_mark:      | :x:    | :x:          |
| Hopper-v3    | :white_check_mark:      | :x:    | :x:          |
| Walker2d-v3    | :white_check_mark:      | :x:    | :x:          |
| Swimmer-v3    | :white_check_mark:      | :x:    | :x:          |
| Humanoid-v3    | :white_check_mark:      | :x:    | :x:          |
| HumanoidStandup-v3    | :white_check_mark:      | :x:    | :x:          |
| Pusher-v3    | :white_check_mark:      | :x:    | :x:          |
| Reacher-v3    | :white_check_mark:      | :x:    | :x:          |
| InvertedDoublePendulum-v3    | :white_check_mark:      | :x:    | :x:          |
| InvertedPendulum-v3    | :white_check_mark:      | :x:    | :x:          |
|  |   |  |  |
| Ant-v5         |    :white_check_mark:     | :x:          |:x:          |
| HalfCheetah-v5    | :white_check_mark:      | :x:    | :x:          |
| Hopper-v5    | :white_check_mark:      | :x:    | :x:          |
| Walker2d-v5    | :white_check_mark:      | :x:    | :x:          |
| Swimmer-v5    | :white_check_mark:      | :x:    | :x:          |
| Humanoid-v5    | :white_check_mark:      | :x:    | :x:          |
| HumanoidStandup-v5    | :white_check_mark:      | :x:    | :x:          |
| Pusher-v5    | :white_check_mark:      | :x:    | :x:          |
| Reacher-v5    | :white_check_mark:      | :x:    | :x:          |
| InvertedDoublePendulum-v5    | :white_check_mark:      | :x:    | :x:          |
| InvertedPendulum-v5    | :white_check_mark:      | :x:    | :x:          |



