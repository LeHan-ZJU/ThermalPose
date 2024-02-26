# from pycocotools.coco import COCO
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # 初始化 COCO API
# dataDir = 'G:/Data/PoseEstimation/COCO'
# dataType = 'val2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# coco = COCO(annFile)

# # 获取所有图像ID
# imgIds = coco.getImgIds()

# # 选择一个图像的ID
# img_id = imgIds[0]

# # 获取该图像的注释
# annIds = coco.getAnnIds(imgIds=img_id)
# anns = coco.loadAnns(annIds)

# # 获取图像文件名并加载图像
# img_info = coco.loadImgs(img_id)[0]
# img_path = '{}/{}/{}'.format(dataDir, dataType, img_info['file_name'])
# image = Image.open(img_path)

# # 创建一个与图像相同大小的全零矩阵
# mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

# # 将实例分割标签绘制在全零矩阵上
# for ann in anns:
#     if 'segmentation' in ann:
#         coco.drawAnn(mask, ann['id'])

# # 将 mask 转换成 PIL Image 对象
# mask_image = Image.fromarray(mask)

# # 保存为 jpg 格式
# mask_image.save('G:/Data/PoseEstimation/COCO/SavedMask.jpg')

# # 显示保存的 mask 图像
# plt.imshow(mask_image)
# plt.axis('off')
# plt.show()

# 导入必要的库
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os

# 设定coco数据集的路径
dataDir = 'G:/Data/PoseEstimation/COCO'
dataType = 'val2017'  # or 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)

# 初始化COCO API
coco=COCO(annFile)

# 获取所有图像的id
imgIds = coco.getImgIds()

# 获取'person'类别的id
catIds = coco.getCatIds(catNms=['person'])

for img_id in imgIds:
    # 获取图像信息
    img_info = coco.loadImgs(ids=img_id)[0]
    img_name = img_info['file_name'].split(".")[0]

    # 获取该图像的所有实例分割标签(只包括'person'类别)
    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    
    # 创建一个空矩阵，用于保存实例分割标签
    mask = np.zeros((img_info['height'],img_info['width']))

    for ann in anns:
        # 用COCO API获取实例分割标签
        m = coco.annToMask(ann)
        mask += m

    # 将实例分割标签转换为255范围的uint8类型
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(os.path.join(dataDir, 'person_masks', img_name + '.jpg'))