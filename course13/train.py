import numpy as np
import paddle
import paddle.fluid as fluid
from cnn import CNN


def test_train(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = fluid.dygraph.base.to_variable(dy_x_data)
        label = fluid.dygraph.base.to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def train_mnist():
    epoch_num = 1
    BATCH_SIZE = 64
    with fluid.dygraph.guard(place=fluid.CPUPlace()):

        cnn = CNN("mnist")
        model, _ = fluid.dygraph.load_persistables("models")
        cnn.load_dict(model)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = fluid.dygraph.base.to_variable(dy_x_data)
                label = fluid.dygraph.base.to_variable(y_data)
                label.stop_gradient = True

                cost, acc = cnn(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                cnn.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))

            cnn.eval()
            test_cost, test_acc = test_train(test_reader, cnn, BATCH_SIZE)
            cnn.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))

        fluid.dygraph.save_persistables(cnn.state_dict(), "models")
        print("checkpoint saved")


train_mnist()
