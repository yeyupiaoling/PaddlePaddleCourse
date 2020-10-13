import re
import uuid
import requests
import os
import numpy
import imghdr
from PIL import Image


# 获取百度图片下载图片
def download_image(key_word, save_name, download_max):
    download_sum = 0
    # 把每个类别的图片存放在单独一个文件夹中
    save_path = 'images' + '/' + save_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while download_sum < download_max:
        download_sum += 1
        str_pn = str(download_sum)
        # 定义百度图片的路径
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
              'word=' + key_word + '&pn=' + str_pn + '&gsm=80&ct=&ic=0&lm=-1&width=0&height=0'
        try:
            # 获取当前页面的源码
            result = requests.get(url, timeout=30).text
            # 获取当前页面的图片URL
            img_urls = re.findall('"objURL":"(.*?)",', result, re.S)
            if img_urls is None or len(img_urls) < 1:
                break
            # 开始下载图片
            for img_url in img_urls:
                # 获取图片内容
                img = requests.get(img_url, timeout=30)
                # 保存图片
                with open(save_path + '/' + str(uuid.uuid1()) + '.jpg', 'wb') as f:
                    f.write(img.content)
                print('正在下载 %s 的第 %d 张图片' % (key_word, download_sum))
                download_sum += 1
                # 下载次数超过指定值就停止下载
                if download_sum >= download_max:
                    break
        except:
            continue
    print('下载完成')


# 删除不是JPEG或者PNG格式的图片
def delete_error_image(father_path):
    image_dirs = os.listdir(father_path)
    for image_dir in image_dirs:
        image_dir = os.path.join(father_path, image_dir)
        # 如果是文件夹就继续获取文件夹中的图片
        if os.path.isdir(image_dir):
            images = os.listdir(image_dir)
            for image in images:
                image = os.path.join(image_dir, image)
                try:
                    # 获取图片的类型
                    image_type = imghdr.what(image)
                    # 如果图片格式不是JPEG同时也不是PNG就删除图片
                    if image_type is not 'jpeg' and image_type is not 'png':
                        os.remove(image)
                        print('已删除：%s' % image)
                        continue
                    # 删除灰度图
                    img = numpy.array(Image.open(image))
                    if len(img.shape) is 2:
                        os.remove(image)
                        print('已删除：%s' % image)
                except:
                    os.remove(image)
                    print('已删除：%s' % image)


if __name__ == '__main__':
    # 定义要下载的图片中文名称和英文名称，ps：英文名称主要是为了设置文件夹名
    key_words = {'西瓜': 'watermelon', '哈密瓜': 'cantaloupe',
                 '樱桃': 'cherry', '苹果': 'apple',
                 '葡萄': 'grape', '梨': 'pear'}
    # 下载图像
    max_sum = 300
    for key_word in key_words:
        save_name = key_words[key_word]
        download_image(key_word, save_name, max_sum)

    # 删除错误图片
    delete_error_image('images/')
