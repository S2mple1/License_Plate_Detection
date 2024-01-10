from Net.colorNet import myNet_ocr_color
import torch
import cv2
import numpy as np
import os
import time


def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    index = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index


def image_processing(img, device):
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, is_color=False):
    input = image_processing(img, device)
    if is_color:
        preds, color_preds = model(input)
        color_preds = torch.softmax(color_preds, dim=-1)
        color_conf, color_index = torch.max(color_preds, dim=-1)
        color_conf = color_conf.item()
    else:
        preds = model(input)
    preds = torch.softmax(preds, dim=-1)
    prob, index = preds.max(dim=-1)
    index = index.view(-1).detach().cpu().numpy()
    prob = prob.view(-1).detach().cpu().numpy()
    newPreds, new_index = decodePlate(index)
    prob = prob[new_index]
    plate = ""
    for i in newPreds:
        plate += plateName[i]
    if is_color:
        return plate, prob, color[color_index], color_conf  # 返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate, prob


def init_model(device, model_path, is_color=False):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    color_classes = 0
    if is_color:
        color_classes = 5  # 颜色类别数
    model = myNet_ocr_color(num_classes=len(plateName), export=True, cfg=cfg, color_num=color_classes)

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    model_path = r"model/plate_rec_color.pth"
    image_path = "imgs/1.jpg"
    testPath = r"Chinese_license_plate_detection_recognition-main/imgs"
    fileList = []
    allFilePath(testPath, fileList)
    is_color = False
    model = init_model(device, model_path, is_color=is_color)
    right = 0
    begin = time.time()

    for imge_path in fileList:
        img = cv2.imread(imge_path)
        if is_color:
            plate, _, plate_color, _ = get_plate_result(img, device, model, is_color=is_color)
            print(plate)
        else:
            plate, _ = get_plate_result(img, device, model, is_color=is_color)
            print(plate, imge_path)
