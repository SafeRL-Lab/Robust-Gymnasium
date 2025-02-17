import argparse

# python main.py --device 2
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    parser = argparse.ArgumentParser(description='Robust env example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="Ant-v4", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=543, metavar='G',
                        help='env seed (default: 543)')
    parser.add_argument('--noise-factor', default="state", metavar='G',
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
    parser.add_argument('--llm-guide-factor', default="state", metavar='G',
                        help='name of the llm guide robust type, e.g., adversary')
    parser.add_argument('--llm-disturb-interval', type=int, default=500, metavar='G',
                        help='llm disturb interval (default: 500)')
    parser.add_argument('--llm-guide-type', default="Non-gauss", metavar='G',
                        help='name of the llm guide robust factor type, e.g., stochastic, uniform')
    parser.add_argument('--uniform-low', type=float, default=0.4, metavar='G',
                        help='unidorm noise low values (default: 0.4)')
    parser.add_argument('--uniform-high', type=float, default=0.4, metavar='G',
                        help='unidorm noise low values (default: 0.4)')
    
    

    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--resume-path", type=str, default=None)

    parser.add_argument('--gravity', type=float, default=9.81, metavar='G',  # 0.04-0.045
                        help='gravity (default: 9.81)')
    parser.add_argument('--wind', type=float, default=0, metavar='G',  # 0.04-0.045
                        help='wind (default: 0)')
    parser.add_argument('--torso-len', type=float, default=0.2, metavar='G',  # 0.04-0.045
                        help='torso_len (default: 9.81)')
    parser.add_argument('--foot-len', type=float, default=0.1, metavar='G',  # 0.04-0.045
                        help='foot_len (default: 0)')

    parser.add_argument('--deter-noise', type=str2bool, default=False,
                        help='deterministic noise (default: False)')

    parser.add_argument('--robust-res', type=str2bool, default=False,
                        help='robust reset (default: True)')




    return parser
