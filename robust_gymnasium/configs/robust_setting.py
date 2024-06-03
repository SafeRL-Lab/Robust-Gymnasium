import argparse


def get_config():
    parser = argparse.ArgumentParser(description='Robust env example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="Ant-v4", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--noise-type', default="gauss", metavar='G',
                        help='name of the algorithm to run')
    parser.add_argument('--noise-mu', type=float, default=0.0, metavar='G',
                        help='noise mean (default: 0.0)')
    parser.add_argument('--noise-sigma', type=float, default=0.005, metavar='G', # 0.04-0.045
                        help='noise variance (default: 0.05)')

    return parser