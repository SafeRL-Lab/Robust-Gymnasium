.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   link back this repository and cite this work.

Environment Disruptor
--------------------------------

**Environment-disruptor**: 
    Recall that a task environment consists of both the internal dynamic model and the external workspace it interacts with, characterized by its transition dynamics :math:`P` and reward function :math:`r`. 
    The environment during training can differ from the real-world environment due to factors such as the sim-to-real gap, human and natural variability, external disturbances, and more. 
    We attribute this potential nonstationarity to an environment-disruptor, which determines the actual environment :math:`(P, r)` the agent is interacting with at any given moment. 
    These dynamics may differ from the nominal environment :math:`(P^0, r^0)` that the agent was originally expected to interact with.


`Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

`Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__