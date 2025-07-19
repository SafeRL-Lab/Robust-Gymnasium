
<!-- 
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/imgs/logo-git.png" width="250"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>
<div align="center">
    <p>Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning</p>
</div>
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/imgs/Robust-RL-Benchmark-logo.jpg" width="100"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>

 <div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/imgs/logo-git-ac.png" width="280"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>
 
## Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning
-->

<div align="center">
  <a href="https://github.com/SafeRL-Lab/Robust-Gymnasium">
    <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/_static/images/logo-git-ac.png" alt="Logo" width="280"> 
  </a>
  
<h1 align="center" style="font-size: 30px;"><strong><em>Robust Gymnasium</em></strong>: A Unified Modular Benchmark for Robust Reinforcement Learning</h1>
<p align="center">
    <a href="https://arxiv.org/pdf/2502.19652">Paper</a>
    ·
    <a href="https://robust-gym.github.io/">Website</a>
    ·
    <a href="https://github.com/SafeRL-Lab/Robust-Gymnasium">Code</a>
    ·
    <a href="https://robust-gymnasium.readthedocs.io/">Tutorial</a>
    ·
    <a href="https://github.com/SafeRL-Lab/Robust-Gymnasium/issues">Issue</a>
  </p>
</div>


 ---

 
 <div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/overview/gif-edit-overview.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>
 

<br/>


This benchmark aims to advance robust reinforcement learning (RL) for real-world applications and domain adaptation. The benchmark provides a comprehensive set of tasks that cover various robustness requirements in the face of uncertainty on state, action, reward and environmental dynamics, and span diverse applications including control, robot manipulations, dexterous hand, and so on (This repository is under actively development. We appreciate any constructive comments and suggestions). 

🔥 **Benchmark Features:** 

- **High Modularity:** It is designed for flexible adaptation to a variety of research needs, featuring high modularity to support a wide range of experiments.
- **Task Coverage:** It provides a comprehensive set of tasks to evaluate robustness across different RL scenarios (at least 170 tasks).
- **High Compatibility:** It can be seamless and compatible with a wide range of existing environments.
- **Support Vectorized Environments:** It can be useful to enable parallel processing of multiple environments for efficient experimentation.
- **Support for New Gym API:** It fully supports the latest standards in Gym API, facilitating easy integration and expansion.
- **LLMs Guide Robust Learning:** Leverage LLMs to set robust parameters (LLMs as adversary policies).

🔥 **Benchmark Tasks:**

- **Robust MuJoCo Tasks:** Tackle complex simulations with enhanced robustness.
- **Robust Box2D Tasks:** Engage with 2D physics environments designed for robustness evaluation.
- **Robust Robot Manipulation Tasks:** Robust robotic manipulation with Kuka and Franka robots.
- **Robust Safety Tasks:** Prioritize safety in robustness evaluation.
- **Robust Android Hand Tasks:** Explore sophisticated hand manipulation challenges in robust settings.
- **Robust Dexterous Tasks:** Advance the robust capabilities in dexterous robotics.
- **Robust Fetch Manipulation Tasks:** Robust object manipulation with Fetch robots.
- **Robust Robot Kitchen Tasks:** Robust manipulation in Kitchen environments with robots.
- **Robust Maze Tasks:** Robust navigation robots.
- **Robust Humanoid Robot Tasks:** Humanoid robot control with robust settings.
- **Robust Multi-Agent Tasks:** Facilitate robust coordination among multiple agents.

Each of these robust tasks incorporates robust elements such as robust observations, actions, reward signals, and dynamics to evaluate the robustness of RL algorithms.

🔥 **Our Vision:**
We hope this benchmark serves as a useful platform for pushing the boundaries of RL in real-world problems --- promoting robustness and domain adaptation ability!

Any suggestions and issues are welcome. If you have any questions, please propose an issue or pull request, or contact us directly via email at shangding.gu@berkeley.edu; we will respond to you in **one week**.

----------
**Content**
<!-- - [Compare with Existing Platforms](#compare-with-existing-platforms) -->
- [Introduction](#introduction)
- [Environments and Tasks](#environments-and-tasks)
- [Disruptor Module for Perturbations](#disruptor-module-for-perturbations)
- [Tutorials](#tutorials)
  * [Installation of the Environment](#installation-of-the-environments)
  * [Quick start](#testing-the-tasks)
- [Selected Demos](#selected-demos)
  * [Robust MuJoCo Tasks](#robust-mujoco-tasks)
  * [Robust MuJoCo Variant Tasks](#robust-mujoco-variant-tasks)
  * [Robust Robot Manipulation Tasks](#robust-robot-manipulation-tasks)
  * [Robust Dexterous Hand and Maze Tasks](#robust-dexterous-hand-and-maze-tasks)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


<!--
----------

## Compare with Existing Platforms
| Robust RL Platforms  | High Modularity   | Task Coverage | Vectorized Environments | High Compatibility  | New Gym API  |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| [Robust Gymnasium](https://github.com/SafeRL-Lab/Robust-Gymnasium)                     | :heavy_check_mark:         | :heavy_check_mark:    | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:  |
| [RLLS](https://github.com/SuReLI/RRLS)                   | :x:         | :x:     |       :x:        | :x:   | :heavy_check_mark:  |

-->



--------------

## Introduction



### Reinforcement Learning against Uncertainty/Perturbation
Reinforcement learning (RL) problems is formulated as that an agent seeks a policy that optimizes the long-term expected return through interacting with an environment. While standard RL has been heavily investigated recently, its use can be significantly hampered in practice due to noise, malicious attacks, the sim-to-real gap, domain generalization requirements, or even a combination of those and more factors. Consequently, in addition to maximizing the cumulative rewards, robustness to unexpected uncertainty/perturbation emerges as another critical goal for RL, especially in high-stakes applications such as robotics, financial investments, autonomous driving, and so on. This leads to a surge of considerations of more robust RL algorithms for different problems, termed as robust RL, including but not limited to single-agent RL, safe RL, and multi-agent RL. 

### A Unified Robust Reinforcement Learning Framework: MDP with Disruption
Robust RL problems typically consists of three modules
* **An agent (a policy):** tries to learn a strategy $\pi$ (a policy) based on the observation from the environment to achieve optimal long-term return
* **An environment/task:** a task that determine the agents' immediate reward $r(\cdot |s,a)$ and the physical or logical dynamics (transition function $`P_t( \cdot | s,a)`$)
* **The disruptor module:** represents the uncertainty/perturbation events that happens during any parts of the interaction process between the agent and environment, with different modes, sources, and frequencies.

We illustrate the framework of robust RL for single-agent problems for instance:
<!-- A figure that shows the modules in our frameworks -->
<div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/_static/images/overview-framework2.png" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>

### Robust-Gymnasium: A Unified Modular Benchmark

<!-- A figure that shows the modules in our frameworks -->
<div align=center>
 <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/docs/_static/images/tasks-illustration2.png" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>
 
This benchmark support various 1) environments/tasks and 2) disruptors （perturbations to the interaction process). This allows users to design and evaluate different algorithms in different application scenarios when encountering diverse uncertainty issues. Switch to the sections below if you want to get a quick glance of which **environments** and **perturbations** that Robust-Gymnasium support.
- [Environments and Tasks](#environments-and-tasks)
- [Disruptor Module for Perturbations](#disruptor-module-for-perturbations)

--------------

## Environments and Tasks

<!-- 
- [x] Robust Single Agent Tasks
- [x] Robust Boxd2d Tasks
- [x] Safe Robust MuJoCo Tasks
- [x] Safe Robust Manipulation Tasks
- [x] Domain Randomization
- [x] Robust Multi-Agent Tasks
- [x] Robust Maze Tasks
- [x] Robust Dexterous Tasks
-->

Tasks: Random, Adversary, Semantic Tasks (Robot Manipulation Tasks).

<details>
<summary><b><big> Robust MuJoCo Tasks </big></b></summary>

| Tasks\Robust type | Robust State | Robust Action | Robust Reward | Robust Dynamics |
|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Ant-v2-v3-v4-v5         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |:white_check_mark:          |
| HalfCheetah-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Hopper-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Walker2d-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Swimmer-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Humanoid-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| HumanoidStandup-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Pusher-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| Reacher-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| InvertedDoublePendulum-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
| InvertedPendulum-v2-v3-v4-v5     | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |:white_check_mark:          |
</details>



<details>
<summary><b><big>  Robust Boxd2d Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| CarRacing-v2       |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| LunarLanderContinuous-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| BipedalWalker-v3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| LunarLander-v3 (Discrete Task)    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
</details>

<details>
<summary><b><big>  Robust Robot Manipulation Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| RobustLift       |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| RobustDoor    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustNutAssembly   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustPickPlace   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustStack   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustWipe   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustToolHang   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustTwoArmLift   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustTwoArmPegInHole  | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustTwoArmHandover   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustTwoArmTransport | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MultiRobustDoor   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
</details>

<details>
<summary><b><big>  Robust Safety Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| RobustSafetyAnt-v4         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| RobustSafetyHalfCheetah-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyHopper-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyWalker2d-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetySwimmer-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyHumanoid-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyHumanoidStandup-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyPusher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustSafetyReacher-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

</details>

<details>
<summary><b><big>  Robust Androit Hand Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| RobustAdroitHandDoor-v1         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| RobustAdroitHandHammer-v1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustAdroitHandPen-v1   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustAdroitHandRelocate-v1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

</details>



<details>
<summary><b><big>  Robust Dexterous Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| RobustHandManipulateEgg_BooleanTouchSensors-v1         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| RobustHandReach-v2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustHandManipulateBlock-v1   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustHandManipulateEgg-v1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustHandManipulatePen-v1   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

</details>

<details>
<summary><b><big>  Robust Fetch Manipulation Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| RobustFetchPush-v3         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| RobustFetchReach-v3   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustFetchSlide-v3   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| RobustFetchPickAndPlace-v3   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

</details>

<details>
<summary><b><big>  Robust Robot Kitchen Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| FrankaKitchen-v1        |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |

</details>

<details>
<summary><b><big>  Robust Maze Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| AntMaze_UMaze-v4        |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| PointMaze_UMaze-v3       |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |

</details>


<details>
<summary><b><big> Robust Multi-Agent Tasks</big></b></summary>

| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| MA-Ant-2x4, 2x4d, 4x2, 4x1         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| MA-HalfCheetah-2x3, 6x1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Hopper-3x1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Walker2d-2x3    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Swimmer-2x1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Humanoid-9\|8    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-HumanoidStandup-v4    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Pusher-3p    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| MA-Reacher-2x1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Many-MA-Swimmer-10x2, 5x4, 6x1, 1x2    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Many-MA-Ant-2x3, 3x1    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| CoupledHalfCheetah-p1p    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
</details>



<details>
<summary><b><big>  Robust Humanoid Tasks</big></b></summary>
  
| Tasks\Robust type | Robust State | Robust Action | Robust Reward |
|:-------------:|:--------------:|:--------------:|:--------------:|
| Robusth1hand-reach-v0         |    :white_check_mark:     | :white_check_mark:          |:white_check_mark:          |
| Robusth1hand-push-v0    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| h1hand-truck-v0   | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |
| Robusth1hand-slide-v0    | :white_check_mark:      | :white_check_mark:    | :white_check_mark:          |

</details>


---------------

## Disruptor Module for Perturbations
Before introducing the disruptor module, we recall that RL problem can be formulated as a process involving several key concepts: an agent, state, action, reward, and an environment. Specifically, at each time $t$, the environment generate a state $s_t$ and a reward $r_t$ and send them to the agent, and the agent choose an action $a_t$ and send back to the environment to generate the next state $s_{t+1}$ conditioned on the current state $s_t$ and the action $a_t$.

Considering this, in this benchmark, we consider extensive potential uncertainty/disturbance/generalizable events that happen in this process (including both training and testing phases) during any places, with any modes, and at any time, summarized in the following table. 

| Perturbation modes\sources | Observed state | Observed reward | Action  | Environment/task |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| Random | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Adversarial | :white_check_mark: | \ | :white_check_mark: | \ |
| Set arbitrarily | \ | \ | \ | :white_check_mark: |
| Semantic Domain shift | \ | \ |  \ | :white_check_mark: |

Those perturbation events can be generally categorized from three different perspectives:
- **Sources:** which component is perturbed/attacked.
  * **Agent's observed state**: The agent observes a noisy/attacked 'state' $\widetilde{s}_t$ (diverge from the real state $s_t$ ) and use it as the input of its policy to determine the action. 
  * **Agent's observed reward**: The agent observes a noisy/attacked 'reward' $\widetilde{r}_t$ (differ from the real immediate reward ($r_t$) obtained from the environment) and construct their policy according to it.
  * **Action**: The action $a_t$ chosen by the agent is contaminated before sent to the environment. Namely, a perturbed action $\widetilde{a}_t$ serves as the input of the environment for the next step.
  * **Environment**: an environment includes both immediate reward function $r$ and dynamic function $P_t$. An agent may interact with a shifted or unstationary environment.
- **Modes:** what kind of perturbation is imposed on.
  * **Random**: the nominal variable will be added by some random noise following some distributions, such as Gaussian, or uniform distribution. This mode can be used to all perturbation sources.
  * **Adversarial**: an adversarial attacker will choose the perturbed output within some admissible set to degrade the agent's performance. This mode can be used to the perturbations towards observation and action.
  * **Set arbitrarily**: An environment can be set to any fixed one within some pre-scribed uncertainty set of the environments.  
  * **Semantic-domain-shifted**: We offer some partially-similar environment/tasks while with some semantic diversity (such as different goals) for domain generalization or transfer learning tasks.
- **Frequency:** when does the perturbation happen. Viewed through the lens of time, the perturbations can happen at different period during training and testing process, even with different frequency.
We provide interactive modes that support step-wise varying interaction between disruptors, agents, and environments. So the user can choose to apply perturbations at any point in the dimension of time in any way. 




<div style="background-color: #d4edda; border-left: 5px solid #00b894; padding: 10px; border-radius: 4px;">
  <strong style="color: #00b894;">&#x1F4A1; Tip</strong><br>
  Not all environments support all kinds of disruptors (perturbations). Please refer to the above section (Environments and Tasks) for more information.
</div>



---------------

## Tutorials

Here, we provide a step-by-step tutorial for users to create and use a domain-shifted/noisy task by choosing any environment/task combined with any uncertainty factor to perturb some original environment, see the [link](https://robust-gymnasium.readthedocs.io/en/latest/).


---------------

<!-- Intro: https://robust-gymnasium.readthedocs.io/en/latest/-->

### Installation of the Environments

1. **Create an environment (requires [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):**
We are currently developing our environments using a Linux system. The operating system version of our server is 20.04.3 LTS.

   Use the following command to create a new Conda environment named `robustgymnasium` with Python 3.11:

   ```bash
   conda create -n robustgymnasium  python=3.11
   ```

   Activate the newly created environment:

   ```bash
   conda activate robustgymnasium
   ```

2. **Install dependency packages:**

   Install the necessary packages using pip. Make sure you are in the project directory where the `setup.py` file is located:

   ```bash
   git clone https://github.com/SafeRL-Lab/Robust-Gymnasium
   cd Robust-Gymnasium
   pip install -r requirements.txt
   pip install -e .
   ```

3. **(Optional) Install with `uv`**

   If you prefer using [`uv`](https://github.com/astral-sh/uv) for faster environment setup:

   ```bash
   pip install uv
   git clone https://github.com/SafeRL-Lab/Robust-Gymnasium
   cd Robust-Gymnasium
   uv venv robustgymnasium --python=3.11
   source robustgymnasium/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```




---------------
### Testing the Tasks

To run the tests, navigate to the `examples` directory and Test. te the test script, e.g.,

```bash
cd examples/robust_action/mujoco/ 
chmod +x test.sh
./test.sh
```

Ensure you follow these steps to set up and test the environment properly. Adjust paths and versions as necessary based on your specific setup requirements.

If you met **some issues**, please check the [existing solutions for the reported issues](https://github.com/SafeRL-Lab/Robust-Gymnasium/issues?q=is%3Aissue+is%3Aclosed), which could help you address your issue.

----------



## Selected Demos


### Robust MuJoCo Tasks
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


### Robust MuJoCo Variant Tasks
<p align="center">
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_vriant_1%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_variant_2%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_variant_3%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>
   
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_variant_4%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_variant_5%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/Ant_variant/Ant_variant_6%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>

  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/HalfCheetah_variant/HalfCheetah_variant_1%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/HalfCheetah_variant/HalfCheetah_variant_2%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/HalfCheetah_variant/HalfCheetah_variant_3%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/> 
  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/HalfCheetah_variant/HalfCheetah_variant_5%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_mujoco_dynamics/HalfCheetah_variant/HalfCheetah_variant_6%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
</p>
<p align="center">
  These demonstrations are Robust MuJoCo variant tasks with robust settings.
</p>






### Robust Robot Manipulation Tasks
<p align="center">
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/multi-robot-door-small.gif" alt="Image 1" width="280" style="margin-right: 10px;"/>  
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Nominal_task%2000_00_00-00_00_30.gif" alt="Image 1" width="310" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_1%2000_00_00-00_00_30.gif" alt="Image 2" width="310" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_2%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>
   
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_3%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_4%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_5%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>

  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_6%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_7%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_8%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/> 
  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_9%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_10%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_11%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_12%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_manipulations/robust_door/Shifted_task_13%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
</p>
<p align="center">
  These demonstrations are from robot manipulation tasks with robust settings.
</p>

### Robust Dexterous Hand and Maze Tasks
<p align="center">
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_andoit_hand/Robust_androit_hand_2%2000_00_00-00_00_30.gif" alt="Image 1" width="310" style="margin-right: 10px;"/>  
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_andoit_hand/Robust_androit_hand_3%2000_00_00-00_00_30.gif" alt="Image 1" width="310" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_andoit_hand/androit_hand_1%2000_00_00-00_00_30.gif" alt="Image 2" width="310" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_andoit_hand/robust_andoit_hand_4%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>
   
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_dexterous_hand/robust_dexterous_hand_1%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_dexterous_hand/robust_dexterous_hand_2%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_dexterous_hand/robust_dexterous_hand_4%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/>

  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_dexterous_hand/robust_dexterous_hand_5%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_dexterous_hand/robust_dexterous_hand_6%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_fetch/robust_fetch_1%2000_00_00-00_00_30.gif" alt="Image 3" width="300" style="margin-right: 10px;"/> 
  
   <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_fetch/robust_fetch_2%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_fetch/robust_fetch_4%2000_00_00-00_00_30.gif" alt="Image 2" width="300" style="margin-right: 10px;"/>
  <img src="https://github.com/SafeRL-Lab/Robust-Gymnasium/blob/main/demos/robust_maze/robust_maze_1%2000_00_00-00_00_30.gif" alt="Image 1" width="300" style="margin-right: 100px;"/>
</p>
<p align="center">
  These demonstrations are from dexterous hand and maze tasks with robust settings.
</p>





---------


## Citation
If you find the repository useful, please cite the study
``` Bash
@article{robustrl2024,
  title={Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning},
  author={Gu, Shangding and Shi, Laixi and Wen, Muning and Jin, Ming and Mazumdar, Eric and Chi, Yuejie and Wierman, Adam and Spanos, Costas},
  journal={ICLR},
  year={2025}
}
```

-------


## Acknowledgments

We thank the contributors from [MuJoCo](https://github.com/google-deepmind/mujoco), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium.git), [Humanoid-bench](https://github.com/carlosferrazza/humanoid-bench/tree/main) and [Robosuite](https://github.com/ARISE-Initiative/robosuite).


