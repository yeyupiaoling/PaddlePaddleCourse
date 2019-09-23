import numpy as np
import paddle.fluid as fluid
import random
import gym
from collections import deque
from paddle.fluid.param_attr import ParamAttr


# 定义一个深度神经网络，通过指定参数名称，用于之后更新指定的网络参数
def DQNetWork(ipt, variable_field):
    fc1 = fluid.layers.fc(input=ipt,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc1'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc1_b'.format(variable_field)))
    fc2 = fluid.layers.fc(input=fc1,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc2'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc2_b'.format(variable_field)))
    out = fluid.layers.fc(input=fc2,
                          size=3,
                          param_attr=ParamAttr(name='{}_fc3'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc3_b'.format(variable_field)))
    return out


# 定义更新参数程序
def _build_sync_target_network():
    # 获取所有的参数
    vars = list(fluid.default_main_program().list_vars())
    # 把两个网络的参数分别过滤出来
    policy_vars = list(filter(lambda x: 'GRAD' not in x.name and 'policy' in x.name, vars))
    target_vars = list(filter(lambda x: 'GRAD' not in x.name and 'target' in x.name, vars))
    policy_vars.sort(key=lambda x: x.name)
    target_vars.sort(key=lambda x: x.name)

    # 从主程序中克隆一个程序用于更新参数
    sync_program = fluid.default_main_program().clone()
    with fluid.program_guard(sync_program):
        sync_ops = []
        for i, var in enumerate(policy_vars):
            sync_op = fluid.layers.assign(policy_vars[i], target_vars[i])
            sync_ops.append(sync_op)
    # 修剪第二个玩了个的参数，完成更新参数
    sync_program = sync_program._prune(sync_ops)
    return sync_program


# 定义输入数据
state_data = fluid.layers.data(name='state', shape=[2], dtype='float32')
action_data = fluid.layers.data(name='action', shape=[1], dtype='int64')
reward_data = fluid.layers.data(name='reward', shape=[], dtype='float32')
next_state_data = fluid.layers.data(name='next_state', shape=[2], dtype='float32')
done_data = fluid.layers.data(name='done', shape=[], dtype='float32')

# 实例化一个游戏环境，参数为游戏名称
# env = gym.make("CartPole-v1")
env = gym.make("MountainCar-v0")
replay_buffer = deque(maxlen=10000)

# 获取网络
state_model = DQNetWork(state_data, 'policy')

# 克隆预测程序
predict_program = fluid.default_main_program().clone()

# 定义损失函数
action_onehot = fluid.layers.one_hot(action_data, 3)
action_value = fluid.layers.elementwise_mul(action_onehot, state_model)
pred_action_value = fluid.layers.reduce_sum(action_value, dim=1)

targetQ_predict_value = DQNetWork(next_state_data, 'target')
best_v = fluid.layers.reduce_max(targetQ_predict_value, dim=1)
best_v.stop_gradient = True
target = reward_data + 1.0 * best_v * (1.0 - done_data)

cost = fluid.layers.square_error_cost(pred_action_value, target)
avg_cost = fluid.layers.reduce_mean(cost)

# 获取更新参数程序
_sync_program = _build_sync_target_network()

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3, epsilon=1e-3)
opt = optimizer.minimize(avg_cost)

# 创建执行器并进行初始化
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义训练的参数
batch_size = 32
num_exploration_episodes = 100
initial_epsilon = 1.0
final_epsilon = 0.01
update_num = 0
epsilon_num = 0
num_episodes = 3000
epsilon = initial_epsilon
# 开始玩游戏
for epsilon_id in range(num_episodes):
    epsilon = max(initial_epsilon * (num_exploration_episodes - epsilon_id) /
                  num_exploration_episodes, final_epsilon)
    # 初始化环境，获得初始状态
    state = env.reset()
    done = False
    ep_reward = 0
    while not done:
        # 显示游戏界面
        # env.render()
        state = np.expand_dims(state, axis=0)
        # epsilon-greedy 探索策略
        if random.random() < epsilon:
            # 以 epsilon 的概率选择随机下一步动作
            action = env.action_space.sample()
        else:
            # 使用模型预测作为结果下一步动作
            action = exe.run(predict_program,
                             feed={'state': state.astype('float32')},
                             fetch_list=[state_model])[0]
            action = np.squeeze(action, axis=0)
            action = np.argmax(action)

        # 让游戏执行动作，获得执行完 动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        next_state, reward, done, info = env.step(action)
        state = next_state

        # 累计每次奖励或者惩罚
        ep_reward += reward
        # 如果游戏结束，输出结束原因
        if done:
            position = next_state[0]
            velocity = next_state[1]
            # print('Pass:%d, epsilon:%f, reward:%d, action:%s' % (epsilon_id, epsilon, ep_reward, next_state))
            if 0.5 <= position <= 0.6 and -0.07 <= velocity <= 0.07:
                print('Pass:%d, 挑战成功, 得分：%d, 奖励:%d, state:%s' % (epsilon_id, ep_reward, reward, next_state))
                ep_reward += 10
            if ep_reward == -200:
                print('Pass:%d, 挑战失败, 原因：操作次数超过200次' % epsilon_id)


        # 记录游戏输出的结果，作为之后训练的数据
        replay_buffer.append((state, action, ep_reward, next_state, done))

        # 如果收集的数据大于Batch的大小，就开始训练
        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                [np.array(a, np.float32) for a in zip(*random.sample(replay_buffer, batch_size))]

            # 更新参数
            if update_num % 200 == 0:
                exe.run(program=_sync_program)
            update_num += 1

            # 调整数据维度
            batch_action = np.expand_dims(batch_action, axis=-1)
            batch_next_state = np.expand_dims(batch_next_state, axis=1)

            # 执行训练
            exe.run(program=fluid.default_main_program(),
                    feed={'state': batch_state,
                          'action': batch_action.astype('int64'),
                          'reward': batch_reward,
                          'next_state': batch_next_state,
                          'done': batch_done})
