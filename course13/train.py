import os
import numpy as np
import paddle
from cnn import CNN


place = paddle.CPUPlace()
paddle.enable_imperative(place)


# 测函数
def test_train(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        # 把数据集拆分为data和label
        dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)
        # 把训练数据转换为动态图所需的Variable类型
        img = paddle.imperative.to_variable(dy_x_data)
        label = paddle.imperative.to_variable(y_data)
        label.stop_gradient = True
        # 获取网络输出
        test_predict = model(img)
        # 获取准确率函数和损失函数
        test_accuracy = paddle.metric.accuracy(input=test_predict, label=label)
        test_loss = paddle.nn.functional.cross_entropy(input=test_predict, label=label)
        test_avg_loss = paddle.mean(test_loss)
        # 保存每次的计算结果
        acc_set.append(float(test_accuracy.numpy()))
        avg_loss_set.append(float(test_avg_loss.numpy()))
    # 求准确率和损失值的平均值
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


BATCH_SIZE = 64
# 获取网络模型
cnn = CNN()
# 如果之前已经保存模型，可以在这里加载模型
if os.path.exists('models/cnn.pdparams'):
    param_dict, _ = paddle.imperative.load("models/cnn")
    # 加载模型中的参数
    cnn.load_dict(param_dict)
# 获取优化方法
momentum = paddle.optimizer.MomentumOptimizer(learning_rate=1e-3,
                                             momentum=0.9,
                                             parameter_list=cnn.parameters())

# 获取训练和测试数据
train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

# 开始训练
for epoch in range(2):
    for batch_id, data in enumerate(train_reader()):
        # 把数据集拆分为data和label
        dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
        # 把训练数据转换为动态图所需的Variable类型
        img = paddle.imperative.to_variable(dy_x_data)
        label = paddle.imperative.to_variable(y_data)
        # 不需要训练label
        label.stop_gradient = True
        # 获取网络输出
        predict = cnn(img)
        # 获取准确率函数和损失函数
        accuracy = paddle.metric.accuracy(input=predict, label=label)
        loss = paddle.nn.functional.cross_entropy(predict, label)
        avg_loss = paddle.mean(loss)
        # 计算梯度
        avg_loss.backward()
        momentum.minimize(avg_loss)
        # 将参数梯度清零以保证下一轮训练的正确性
        cnn.clear_gradients()
        # 打印一次信息
        if batch_id % 100 == 0:
            print(
                "Epoch:%d, Batch:%d, Loss:%f, Accuracy:%f" % (epoch, batch_id, avg_loss.numpy(), accuracy.numpy()))
    # 开始执行测试
    cnn.eval()
    test_cost, test_acc = test_train(test_reader, cnn, BATCH_SIZE)
    # 准备重新恢复训练
    cnn.train()
    print("Test:%d, Loss:%f, Accuracy:%f" % (epoch, test_cost, test_acc))

    if not os.path.exists('models'):
        os.makedirs('models')
    # 保存模型
    paddle.imperative.save(state_dict=cnn.state_dict(), model_path="models/cnn")
