# 导入VisualDL的包
from visualdl import LogWriter

# 创建一个LogWriter，logdir参数是指定存放数据的路径，
writer = LogWriter(logdir="./random_log")

# 读取数据
for step in range(1000):
    # 在测试分类下创建标量数据1
    writer.add_scalar(tag="测试/数据1", step=step, value=step * 1. / 1000)
    # 在测试分类下创建标量数据2
    writer.add_scalar(tag="测试/数据2", step=step, value=1. - step * 1. / 1000)
