import argparse


def get_config():
    parser = argparse.ArgumentParser(description='Robust env example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="HalfCheetah-v5s", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=543, metavar='G',
                        help='env seed (default: 543)')
    parser.add_argument('--noise-factor', default="Non-state", metavar='G',
                        help='name of noise factor, such as state, action, reward, cost, robust_force (dynamics), robust_shape (mass), dynamics')
    parser.add_argument('--noise-type', default="gauss", metavar='G',
                        help='name of the noise type, e.g., gauss, shift, uniform, Non_stationary')
    parser.add_argument('--noise-mu', type=float, default=0.0, metavar='G',
                        help='noise mean (default: 0.0)')
    parser.add_argument('--noise-sigma', type=float, default=0.005, metavar='G', # 0.04-0.045
                        help='noise variance (default: 0.05)')
    parser.add_argument('--robust-force-mu', type=float, default=0.0, metavar='G',
                        help='noise mean (default: 0.0)')
    parser.add_argument('--robust-force-sigma', type=float, default=0.005, metavar='G',  # 0.04-0.045
                        help='noise variance (default: 0.05)')
    parser.add_argument('--robust-shape-mu', type=float, default=0.0, metavar='G',
                        help='noise mean (default: 0.0)')
    parser.add_argument('--robust-shape-sigma', type=float, default=0.05, metavar='G',  # 0.04-0.045
                        help='noise variance (default: 0.05)')
    parser.add_argument('--uniform-min-val', type=float, default=0.05, metavar='G',  # 0.04-0.045
                        help='robust uniform (default: 0.05)')
    parser.add_argument('--uniform-max-val', type=float, default=0.55, metavar='G',  # 0.04-0.045
                        help='robust uniform (default: 0.05)')
    parser.add_argument('--noise-shift', type=float, default=0.005, metavar='G',  # 0.04-0.045
                        help='noise shift (default: 0.05)')
    parser.add_argument('--env-robosuite-robust', default="Lift-Semantic", metavar='G',
                        help='Semantic tasks: name of the robosuite environment to run, Door-Semantic')
    parser.add_argument('--door-table-height', type=float, default=0.8, metavar='G',
                        help='robosuite door task: table height (default: 0.8)')
    parser.add_argument('--door-robot-table-distance', type=float, default=0.005, metavar='G',
                        help='robosuite door task: the distance between robot and table  (default: -0.55)')
    parser.add_argument('--llm-guide', default="Non-adversary", metavar='G',
                        help='name of the llm guide robust type, e.g., adversary')
    parser.add_argument('--llm-disturb-interval', type=int, default=500, metavar='G',
                        help='llm disturb interval (default: 500)')
    
    parser.add_argument('--gravity', type=float, default=9.81, metavar='G',  # 0.04-0.045
                        help='gravity (default: 9.81)')
    parser.add_argument('--wind', type=float, default=0, metavar='G',  # 0.04-0.045
                        help='wind (default: 0)')
    parser.add_argument('--robust-res', type=bool, default=False,
                        help='robust reset (default: True)')

    parser.add_argument("--task", type=str, default="HalfCheetah-v5s")
    parser.add_argument("--resume-path", type=str, default=None)




    return parser
