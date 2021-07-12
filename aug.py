from lxml import etree
from os.path import basename, split, join, dirname
from util import *
import xml.etree.ElementTree as ET
import gdalTools

def find_str(filename):
    if 'train' in filename:
        return dirname(filename[filename.find('train'):])
    else:
        return dirname(filename[filename.find('val'):])


def convert_all_boxes(shape, anno_infos, yolo_label_txt_dir):
    height, width, n = shape
    label_file = open(yolo_label_txt_dir, 'w')
    for anno_info in anno_infos:
        target_id, x1, y1, x2, y2 = anno_info
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)
        label_file.write(str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    check_dir(crop_save_dir)
    crop_img_save_dir = join(crop_save_dir, basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def copysmallobjects(image_dir, label_dir, save_base_dir, save_crop_base_dir=None,
                     save_annoation_base_dir=None):
    image = cv2.imread(image_dir)

    labels = read_label_txt(label_dir)
    if len(labels) == 0: return
    rescale_labels = rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    all_boxes = []

    for idx, rescale_label in enumerate(rescale_labels):

        all_boxes.append(rescale_label)
        # 目标的长宽
        rescale_label_height, rescale_label_width = rescale_label[4] - rescale_label[2], rescale_label[3] - \
                                                    rescale_label[1]

        if (issmallobject((rescale_label_height, rescale_label_width), thresh=64 * 64) and rescale_label[0] == '1'):
            roi = image[rescale_label[2]:rescale_label[4], rescale_label[1]:rescale_label[3]]

            new_bboxes = random_add_patches(rescale_label, rescale_labels, image.shape, paste_number=2, iou_thresh=0.2)
            count = 0

            # 将新生成的位置加入到label,并在相应位置画出物体
            for new_bbox in new_bboxes:
                count += 1
                all_boxes.append(new_bbox)
                cl, bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], \
                                                                   new_bbox[4]
                try:
                    if (count > 1):
                        roi = flip_bbox(roi)
                    image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                except ValueError:
                    continue

    dir_name = find_str(image_dir)
    save_dir = join(save_base_dir, dir_name)
    check_dir(save_dir)
    yolo_txt_dir = join(save_dir, basename(image_dir.replace('.jpg', '_augment.txt')))
    cv2.imwrite(join(save_dir, basename(image_dir).replace('.jpg', '_augment.jpg')), image)
    convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)


def GaussianBlurImg(image):
    # 高斯模糊
    ran = random.randint(0, 9)
    if ran % 2 == 1:
        image = cv2.GaussianBlur(image, ksize=(ran, ran), sigmaX=0, sigmaY=0)
    else:
        pass
    return image


def suo_fang(image, area_max=1100, area_min=700):
    # 改变图片大小
    height, width, channels = image.shape

    while (height*width) > area_max:
        image = cv2.resize(image, (int(width * 0.95),int(height * 0.95)))
        height, width, channels = image.shape
        height, width = int(height*0.95), int(width*0.95)

    while (height*width) < area_min:
        image = cv2.resize(image, (int(width * 1.2), int(height * 1.2)))
        height, width, channels = image.shape
        height, width = int(height*1.2), int(width*1.2)

    return image


def parse_xml(path, class_dict):
    tree = ET.parse(path)
    root = tree.findall('object')
    class_list = []
    boxes_list = []
    difficult_list = []
    for sub in root:
        xmin = float(sub.find('bndbox').find('xmin').text)
        xmax = float(sub.find('bndbox').find('xmax').text)
        ymin = float(sub.find('bndbox').find('ymin').text)
        ymax = float(sub.find('bndbox').find('ymax').text)
        # if ymax > 915 or xmax > 1044 or xmin < 0 or ymin < 0:
        #     print(xmin, ymin, xmax, ymax, path)
        boxes_list.append([xmin, ymin, xmax, ymax])
        class_list.append(class_dict[sub.find('name').text])
        difficult_list.append(int(sub.find('difficult').text))
    return np.array(class_list), np.array(boxes_list).astype(np.int32)


def copysmallobjects2(image_dir, labels, boxes, save_base_dir, small_img_dir, class_dict):
    # image = cv2.imread(image_dir)
    im_proj, im_geotrans, im_width, im_height, image = gdalTools.read_img(image_dir)
    image = image.transpose((1, 2, 0))
    if len(labels) == 0:
        return
    rescale_labels = []
    for label, box in zip(labels, boxes):
        rescale_labels.append([str(label), int(box[0]), int(box[1]), int(box[2]), int(box[3])])

    all_boxes = []
    for _, rescale_label in enumerate(rescale_labels):
        all_boxes.append(rescale_label)

    for small_img_dirs in small_img_dir:
        image_bbox = cv2.imread(small_img_dirs)
        small_class = os.path.split(small_img_dirs)[-1].split('_')[0]
        #roi = image_bbox
        try:
             # roi = suo_fang(image_bbox, area_max=10000, area_min=1000)
            roi = image_bbox
        except:
            print(small_img_dirs)
            continue

        new_bboxes = random_add_patches2(roi.shape, small_class, rescale_labels, image.shape, paste_number=1, iou_thresh=0)
        count = 0
        for new_bbox in new_bboxes:
            count += 1

            cl, bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], \
                                                               new_bbox[4]
            #roi = GaussianBlurImg(roi)  # 高斯模糊
            height, width, channels = roi.shape
            center = (int(width / 2), int(height / 2))
            #ran_point = (int((bbox_top+bbox_bottom)/2),int((bbox_left+bbox_right)/2))
            mask = 255 * np.ones(roi.shape, roi.dtype)

            try:
                if count > 1:
                    roi = flip_bbox(roi)

                image[bbox_top:bbox_bottom, bbox_left:bbox_right] = cv2.seamlessClone(roi, image[bbox_top:bbox_bottom, bbox_left:bbox_right],
                                                                                      mask, center, cv2.NORMAL_CLONE)
                all_boxes.append(new_bbox)
                rescale_labels.append(new_bbox)
            except ValueError:
                print("---")
                continue

    dir_name = find_str(image_dir)
    save_dir = join(save_base_dir, dir_name)
    check_dir(save_dir)
    # cv2.imwrite(join(save_dir, basename(image_dir).replace('.jpg', '_augment.jpg')), image)
    outfileName = os.path.join(save_dir, basename(image_dir))
    gdalTools.write_img(outfileName, im_proj, im_geotrans, image.transpose((2, 0, 1)))
    # ann = GEN_Annotations(image_dir)
    # ann.set_size(image.shape[0], image.shape[1], image.shape[2])
    # for anno_info in all_boxes:
    #     target_id, x1, y1, x2, y2 = anno_info
    #     label_name = '1'
    #     ann.add_pic_attr(label_name, x1, y1, x2, y2)
    # ann.savefile(xml_dir)


def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]


class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "source")

        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child5.text = "Unknown"

        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        child7 = etree.SubElement(child3, "flickrid")
        child7.text = "35435"

    def set_size(self, witdh, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

