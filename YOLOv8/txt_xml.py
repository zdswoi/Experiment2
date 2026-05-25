from xml.dom.minidom import Document
import os
import cv2


# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径 "0","3","7","10","16","20","40","70","100"
def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
    """
    dic = {  '0': "6-blank",
    '1': "6-CTC-30",
    '2': "6-CTC-60",
    '3': "6-DOX-30",
    '4': "6-DOX-60",
    '5': "6-ENR-30",
    '6': "6-ENR-60",
    '7': "6-FTD-30",
    '8': "6-FTD-60",
    '9': "6-FZD-30",
    '10': "6-FZD-60",
    '11': "6-LEV-30",
    '12': "6-LEV-60",
    '13': "6-NFZ-30",
    '14': "6-NFZ-60",
    '15': "6-NOR-30",
  '16': "6-NOR-60",
  '17': "6-OTC-30",
  '18': "6-OTC-60",
  '19': "6-TC-30",
  '20': "6-TC-60",
  '21': "8-blank",
  '22': "8-CTC-30",
  '23': "8-CTC-60",
  '24': "8-DOX-30",
  '25': "8-DOX-60",
  '26': "8-ENR-30",
  '27': "8-ENR-60",
  '28': "8-FTD-30",
  '29': "8-FTD-60",
  '30': "8-FZD-30",
  '31': "8-FZD-60",
  '32': "8-LEV-30",
  '33': "8-LEV-60",
  '34': "8-NFZ-30",
  '35': "8-NFZ-60",
  '36': "8-NOR-30",
  '37': "8-NOR-60",
  '38': "8-OTC-30",
  '39': "8-OTC-60",
  '40': "8-TC-30",
  '41': "8-TC-60",
  '42': "6-FTD-5",
  '43': "6-FTD-10",
  '44': "6-FTD-20",
  '45': "6-FTD-40",
  '46': "6-FTD-50",
  '47': "6-FTD-70",
  '48': "6-FTD-80",
  '49': "6-FTD-90",
  '50': "6-LEV-5",
  '51': "6-LEV-10",
  '52': "6-LEV-15",
  '53': "6-LEV-20",
  '54': "6-LEV-25",
  '55': "6-LEV-35",
  '56': "6-OTC-10",
  '57': "6-OTC-20",
  '58': "6-OTC-40",
  '59': "6-OTC-50",
  '60': "6-OTC-70",
  '61': "6-OTC-80",
  '62': "6-OTC-90",
  '63': "8-FTD-10",
  '64': "8-FTD-20",
  '65': "8-FTD-40",
  '66': "8-FTD-50",
  '67': "8-FTD-70",
  '68': "8-LEV-5",
  '69': "8-LEV-10",
  '70': "8-LEV-20",
  '71': "8-LEV-40",
  '72': "8-LEV-50",
  '73': "8-LEV-70",
  '74': "8-LEV-80",
  '75': "8-LEV-90",
  '76': "8-OTC-5",
  '77': "8-OTC-10",
  '78': "8-OTC-20",
  '79': "8-OTC-40",
  '80': "8-OTC-50",
  '81': "8-OTC-70"
           }
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        if "classes" in name:
            continue
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


if __name__ == "__main__":
    picPath = "datasets/VOC20071/JPEGImages/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = "datasets/VOC20071/YOLOLabels/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = "datasets/VOC20071/Annotations/"  # xml文件保存路径，后面的/一定要带上
    makexml(picPath, txtPath, xmlPath)
