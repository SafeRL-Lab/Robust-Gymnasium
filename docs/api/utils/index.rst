.. Robust Gymnasium documentation master file, created by
   sphinx-quickstart on Thu Nov 14 19:51:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Utils
--------------------------------

.. code-block:: python

    import argparse

    # Helper function to parse boolean arguments from command line
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Function to define and retrieve the configuration parser
    def get_config():
        parser = argparse.ArgumentParser(description='Robust env example')

        # General environment settings
        parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                            help='Discount factor (default: 0.995)')
        parser.add_argument('--env-name', default="Ant-v4", metavar='G',
                            help='Name of the environment to run')
        parser.add_argument('--seed', type=int, default=543, metavar='G',
                            help='Environment seed (default: 543)')
        parser.add_argument('--noise-factor', default="Non-state", metavar='G',
                            help='Noise factor (e.g., state, action, reward, cost, robust_force, robust_shape)')
        parser.add_argument('--noise-type', default="gauss", metavar='G',
                            help='Noise type (e.g., gauss, shift, uniform, Non_stationary)')
        parser.add_argument('--noise-mu', type=float, default=0.0, metavar='G',
                            help='Noise mean (default: 0.0)')
        parser.add_argument('--noise-sigma', type=float, default=0.005, metavar='G',
                            help='Noise variance (default: 0.005)')
        parser.add_argument('--robust-force-mu', type=float, default=0.0, metavar='G',
                            help='Robust force mean (default: 0.0)')
        parser.add_argument('--robust-force-sigma', type=float, default=0.005, metavar='G',
                            help='Robust force variance (default: 0.005)')
        parser.add_argument('--robust-shape-mu', type=float, default=0.0, metavar='G',
                            help='Robust shape mean (default: 0.0)')
        parser.add_argument('--robust-shape-sigma', type=float, default=0.05, metavar='G',
                            help='Robust shape variance (default: 0.05)')
        parser.add_argument('--uniform-min-val', type=float, default=0.05, metavar='G',
                            help='Uniform distribution minimum value (default: 0.05)')
        parser.add_argument('--uniform-max-val', type=float, default=0.55, metavar='G',
                            help='Uniform distribution maximum value (default: 0.55)')
        parser.add_argument('--noise-shift', type=float, default=0.005, metavar='G',
                            help='Noise shift (default: 0.005)')
        parser.add_argument('--env-robosuite-robust', default="Lift-Semantic", metavar='G',
                            help='Robosuite semantic task (e.g., Door-Semantic)')
        parser.add_argument('--door-table-height', type=float, default=0.8, metavar='G',
                            help='Robosuite door task table height (default: 0.8)')
        parser.add_argument('--door-robot-table-distance', type=float, default=0.005, metavar='G',
                            help='Robosuite door task distance between robot and table (default: 0.005)')
        parser.add_argument('--llm-guide', default="Non-adversary", metavar='G',
                            help='LLM guide robust type (e.g., adversary)')
        parser.add_argument('--llm-disturb-interval', type=int, default=500, metavar='G',
                            help='LLM disturb interval (default: 500)')

        # Task and training settings
        parser.add_argument("--task", type=str, default="Ant-v4",
                            help='Task to be performed')
        parser.add_argument("--resume-path", type=str, default=None,
                            help='Path to resume a saved model')

        # Non-stationary settings
        parser.add_argument('--gravity', type=float, default=9.81, metavar='G',
                            help='Gravity (default: 9.81)')
        parser.add_argument('--wind', type=float, default=0, metavar='G',
                            help='Wind (default: 0)')
        parser.add_argument('--torso-len', type=float, default=0.2, metavar='G',
                            help='Torso length (default: 0.2)')
        parser.add_argument('--foot-len', type=float, default=0.1, metavar='G',
                            help='Foot length (default: 0.1)')
        parser.add_argument('--deter-noise', type=str2bool, default=False,
                            help='Deterministic noise (default: False)')
        parser.add_argument('--robust-res', type=str2bool, default=False,
                            help='Robust reset (default: False)')
        parser.add_argument('--nonstationary-setting', type=str, default="normal_setting",
                            help='Non-stationary setting (default: normal_setting)')

        # Safety Robust RL settings
        parser.add_argument('--algo', default="PCRPO_4S_G_V0", metavar='G',
                            help='Algorithm to run (default: PCRPO_4S_G_V0)')
        parser.add_argument('--cost-limit', type=float, default=0.04, metavar='G',
                            help='Cost limit (default: 0.04)')
        parser.add_argument('--slack-bound', type=float, default=0.005, metavar='G',
                            help='Slack bound (default: 0.005)')
        parser.add_argument('--sample-penaltyDec', type=float, default=0.7, metavar='G',
                            help='Penalty decrease factor (default: 0.7)')
        parser.add_argument('--sample-penaltyInc', type=float, default=1.2, metavar='G',
                            help='Penalty increase factor (default: 1.2)')
        parser.add_argument('--exps-epoch', type=int, default=500, metavar='G',
                            help='Number of epochs for experiments (default: 500)')

        return parser


.. `Github <https://github.com/SafeRL-Lab/Robust-Gymnasium>`__

.. `Contribute to the Docs <https://github.com/PKU-Alignment/safety-gymnasium/blob/main/CONTRIBUTING.md>`__