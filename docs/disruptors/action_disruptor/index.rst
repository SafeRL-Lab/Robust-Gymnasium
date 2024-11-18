.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Action Disruptor
--------------------------------

**Action-disruptor**: 
    The real action :math:`a_t` chosen by the agent may be altered before or during execution in the environment due to implementation inaccuracies or system malfunctions. 
    The action-disruptor models this perturbation, outputting a perturbed action :math:`\tilde{a}_t = D_a(a_t)`, which is then executed in the environment for the next step.


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__