import paddle.fluid as fluid


# 定义VGG16神经网络
def vgg16(input, class_dim=1000):
    def conv_block(conv, num_filter, groups):
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=3,
                                       stride=1,
                                       padding=1,
                                       act='relu')
        return fluid.layers.pool2d(input=conv, pool_size=2, pool_type='max', pool_stride=2)

    conv1 = conv_block(input, 64, 2)
    conv2 = conv_block(conv1, 128, 2)
    conv3 = conv_block(conv2, 256, 3)
    conv4 = conv_block(conv3, 512, 3)
    conv5 = conv_block(conv4, 512, 3)

    fc1 = fluid.layers.fc(input=conv5, size=512)
    dp1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=dp1, size=512)
    bn1 = fluid.layers.batch_norm(input=fc2, act='relu')
    fc2 = fluid.layers.dropout(x=bn1, dropout_prob=0.5)
    out = fluid.layers.fc(input=fc2, size=class_dim, act='softmax')

    return out
