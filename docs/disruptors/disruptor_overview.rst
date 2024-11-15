.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Overview of Disruptor Modules
===============================================

Before introducing the disruptor module, we recall that the RL problem can be formulated as a process involving several key concepts: an agent, state, action, reward, and an environment. Specifically, at each time :math:`t`, the environment generates a state :math:`s_t` and a reward :math:`r_t`, and sends them to the agent, and the agent chooses an action :math:`a_t` and sends it back to the environment to generate the next state :math:`s_{t+1}` conditioned on the current state :math:`s_t` and the action :math:`a_t`.

Considering this, in this benchmark, we consider extensive potential uncertainty/disturbance/generalizable events that happen in this process (including both training and testing phases) during any places, with any modes, and at any time, summarized in the following table.

.. list-table:: 
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Perturbation modes\sources
     - Observed state
     - Observed reward
     - Action
     - Environment/task
   * - Random
     - ✅
     - ✅
     - ✅
     - ✅
   * - Adversarial
     - ✅
     - ✅
     - \\
     - \\
   * - Set arbitrarily
     - \\
     - \\
     - \\
     - ✅
   * - Semantic Domain shift
     - \\
     - \\
     - \\
     - ✅

Those perturbation events can be generally categorized from three different perspectives:

- **Sources**: which component is perturbed/attacked.
  - *Agent's observed state*: The agent observes a noisy/attacked "state" :math:`\tilde{s_t}` (diverging from the real state :math:`s_t`) and uses it as the input of its policy to determine the action.
  - *Agent's observed reward*: The agent observes a noisy/attacked "reward" :math:`\tilde{r_t}` (differing from the real immediate reward :math:`r_t` obtained from the environment) and constructs their policy according to it.
  - *Action*: The action :math:`a_t` chosen by the agent is contaminated before being sent to the environment. Namely, a perturbed action :math:`\tilde{a_t}` serves as the input of the environment for the next step.
  - *Environment*: An environment includes both immediate reward function :math:`r` and dynamic function :math:`p_t`. An agent may interact with a shifted or nonstationary environment.

- **Modes**: what kind of perturbation is imposed on.
  - *Random*: The nominal variable will be added by some random noise following some distributions, such as Gaussian or uniform distribution. This mode can be used to all perturbation sources.
  - *Adversarial*: An adversarial attacker will choose the perturbed output within some admissible set to degrade the agent's performance. This mode can be used for the perturbations towards observation and action.
  - *Set arbitrarily*: An environment can be set to any fixed one within some pre-scribed uncertainty set of the environments.
  - *Semantic-domain-shifted*: We offer some partially-similar environment/tasks while with some semantic diversity (such as different goals) for domain generalization or transfer learning tasks.

- **Frequency**: when does the perturbation happen. Viewed through the lens of time, the perturbations can happen at different periods during training and testing process, even with different frequency. We provide interactive modes that support step-wise varying interaction between disruptors, agents, and environments. So the user can choose to apply perturbations at any point in the dimension of time in any way.



`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__