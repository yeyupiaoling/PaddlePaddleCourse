import numpy as np
import paddle.fluid as fluid
from paddle.dataset.image import load_image, simple_transform
import paddle
try:
    # 兼容PaddlePaddle2.0
    paddle.enable_static()
except:
    pass


# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'models/infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=save_path,
                                             executor=exe)


# 预处理图片
def load_data(file):
    img = load_image(file)
    img = simple_transform(img, 256, 224, is_train=False,
                           mean=[103.94, 116.78, 123.68])
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    return img


# 获取图片数据
img = load_data('image/image_00001.jpg')

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)

# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

print('预测结果标签为：%d， 实际标签为：%d， 概率为：%f'
      % (lab, 76, result[0][0][lab]))
