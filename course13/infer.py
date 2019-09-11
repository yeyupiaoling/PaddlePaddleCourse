import paddle.fluid as fluid
from PIL import Image
import numpy as np
from cnn import CNN

with fluid.dygraph.guard(place=fluid.CPUPlace()):
    # 获取网络结构
    cnn_infer = CNN("models")
    # 加载模型参数
    model, _ = fluid.dygraph.load_persistables("models")
    # 把参数加载到网络中
    cnn_infer.load_dict(model)

    # 开始执行预测
    cnn_infer.eval()

    #
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im


    tensor_img = load_image('image/infer_3.png')

    results = cnn_infer(fluid.dygraph.base.to_variable(tensor_img))
    lab = np.argsort(results.numpy())
    print("Inference result of image/infer_3.png is: %d" % lab[0][-1])
    print(results.numpy()[0][lab[0][-1]])
