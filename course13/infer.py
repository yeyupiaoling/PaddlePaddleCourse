import paddle
from PIL import Image
import numpy as np
from cnn import CNN

place = paddle.CPUPlace()
paddle.enable_imperative(place)

# 获取网络结构
cnn_infer = CNN()
# 加载模型参数
param_dict, _ = paddle.imperative.load("models/cnn")
# 把参数加载到网络中
cnn_infer.load_dict(param_dict)
# 开始执行预测
cnn_infer.eval()


# 预处理数据
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# 获取预测数据
tensor_img = load_image('image/infer_3.png')
# 执行预测
results = cnn_infer(paddle.imperative.to_variable(tensor_img))
# 安装概率从大到小排序标签
lab = np.argsort(-results.numpy())
print("infer_3.png 预测的结果为: %d" % lab[0][0])
