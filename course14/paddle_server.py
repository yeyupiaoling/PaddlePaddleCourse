import os
import cv2
import numpy as np
import uuid
from PIL import Image
from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import create_paddle_predictor as PaddlePredictor

app = Flask(__name__)
# 允许跨越访问
CORS(app)


# 根路径，返回一个字符串
@app.route('/hello')
def hello_world():
    return 'Welcome to PaddlePaddle'


# 上传文件
@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['img']
    # 设置保存路径
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + "." +
                            secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    return 'success, save path: ' + img_path


# 配置预测器信息，指定模型路径
config = AnalysisConfig('models')
# 设置使用CPU
config.disable_gpu()
# 创建预测器
predictor = PaddlePredictor(config)


def load_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def fake_input(img):
    image = PaddleTensor()
    image.name = "image"
    image.shape = img.shape
    image.dtype = PaddleDType.FLOAT32
    image.data = PaddleBuf(img.flatten().tolist())
    return [image]


@app.route('/infer', methods=['POST'])
def infer():
    f = request.files['img']
    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # 获取图像数据
    inputs = fake_input(load_image(img))
    # 执行预测
    outputs = predictor.run(inputs)
    # 获取预测概率值
    result = outputs[0].data.float_data()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[-1]
    names = ["苹果", "哈密瓜", "樱桃", "葡萄", "梨", "西瓜"]
    # 打印和返回预测结果
    r = '{"label":%d, "name":"%s", "possibility":%f}' % (lab, names[lab], result[lab])
    print(r)
    return r


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    # 启动服务，并指定端口号
    app.run(host='0.0.0.0', port=5000)
