import os
import argparse
from alphabets import plate_chr


def allFileList(rootfile, allFile):
    folder = os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile, temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName, allFile)


def is_str_right(plate_name):
    for str_ in plate_name:
        if str_ not in palteStr:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="datasets/plate/val/",
                        help='source')
    parser.add_argument('--label_file', type=str, default='datasets/val.txt', help='model.pt path(s)')

    opt = parser.parse_args()
    rootPath = opt.image_path
    labelFile = opt.label_file
    palteStr = plate_chr
    print(len(palteStr))
    plateDict = {}
    for i in range(len(list(palteStr))):
        plateDict[palteStr[i]] = i
    fp = open(labelFile, "w", encoding="utf-8")
    file = []
    allFileList(rootPath, file)
    picNum = 0
    for jpgFile in file:
        print(jpgFile)
        jpgName = os.path.basename(jpgFile)
        name = jpgName.split("_")[0]
        if " " in name:
            continue
        labelStr = " "
        if not is_str_right(name):
            continue
        strList = list(name)
        for i in range(len(strList)):
            labelStr += str(plateDict[strList[i]]) + " "
        picNum += 1
        fp.write(jpgFile + labelStr + "\n")
    fp.close()
