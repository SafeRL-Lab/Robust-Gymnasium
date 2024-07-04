import numpy as np
from robust_gymnasium.envs.robust_ma_mujoco.mamujoco_v1 import get_parts_and_edges
from robust_gymnasium.envs.robust_ma_mujoco import mujoco_multi # mamujoco_v1
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
unpartioned_nodes, edges, global_nodes = get_parts_and_edges('Ant', None)
print("unpartioned_nodes---:", unpartioned_nodes)

partioned_nodes = [(unpartioned_nodes[0][0],), (unpartioned_nodes[0][1],), (unpartioned_nodes[0][2],), (unpartioned_nodes[0][3],), (unpartioned_nodes[0][4],), (unpartioned_nodes[0][5],), (unpartioned_nodes[0][6],), (unpartioned_nodes[0][7],)]
my_agent_factorization = {"partition": partioned_nodes, "edges": edges, "globals": global_nodes}
env = mujoco_multi.MultiAgentMujocoEnv('Ant', '8x1', agent_factorization=my_agent_factorization, render_mode="human")

# MA-Ant-2x4, 2x4d, 4x2, 4x1	✅	✅	✅
# MA-HalfCheetah-2x3, 6x1	✅	✅	✅
# MA-Hopper-3x1	✅	✅	✅
# MA-Walker2d-2x3	✅	✅	✅
# MA-Swimmer-2x1	✅	✅	✅
# MA-Humanoid-9|8	✅	✅	✅
# MA-HumanoidStandup-v4	✅	✅	✅
# MA-Pusher-3p	✅	✅	✅
# MA-Reacher-2x1	✅	✅	✅
# Many-MA-Swimmer-10x2, 5x4, 6x1, 1x2	✅	✅	✅
# Many-MA-Ant-2x3, 3x1	✅	✅	✅
# CoupledHalfCheetah-p1p

observation, info = env.reset(seed=42)
action = env.single_agent_env.action_space.sample()
assert (
        action == env.map_local_actions_to_global_action(
            env.map_global_action_to_local_actions(action)
        )
    ).all()
actions = []
n_actions = max([len(l) for l in my_agent_factorization['partition']])

for agent_id in range(8):
    avail_actions = [1.] # env.get_avail_agent_actions(agent_id)
    avail_actions_ind = np.nonzero(avail_actions)[0]
    action = np.random.uniform(-1.0, 1.0, n_actions)
    actions.append(action)


re_actions = {'agent_0': [-0.34627497], 'agent_1': [0.13449878], 'agent_2': [-0.02019556], 'agent_3': [0.5044606], 'agent_4': [0.8911963], 'agent_5': [-0.91603947], 'agent_6': [-0.8271199], 'agent_7': [0.7132667]}
# re_actions = {'agent_0': [-0.34627497], 'agent_1': [0.13449878], 'agent_2': [-0.02019556], 'agent_3': [0.5044606], 'agent_4': [0.8911963], 'agent_5': [-0.91603947]}


for i in range(1000):
    # action = env.action_space.sample()
    robust_input = {
        "action": re_actions,
        "robust_type": "state",
        "robust_config": args,
    }
    observation, reward, terminated, truncated, info = env.step(re_actions)
    # print("reward----------:", reward)
    # print("observation------:", observation)
    env.render() # render environments
    if terminated or truncated:
        observation, info = env.reset()
env.close()