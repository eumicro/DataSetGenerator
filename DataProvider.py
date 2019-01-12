import numpy as np
from pathlib import Path
import tensorflow as tf
from DataModel import Region
from xml.dom import minidom
import os
import cv2
from PIL import Image
import hashlib
import io
import binascii
from object_detection.utils import dataset_util

class Label:
    def __init__(self,name,id,x1,y1,x2,y2):
        self.name = str(name)
        self.id = int(id)
        self.p1 = (int(x1),int(y1))
        self.p2 = (int(x2),int(y2))
        self.region = Region(x1,y1,x2,y2,label=name)
    def getRegion(self):
        return self.region

class TSCLabeledDataProvider:
    def __init__(self,dtsdb_path):
        self.folder_path =  Path(dtsdb_path)
        self.file_path = Path(self.folder_path.as_posix()+ "gt.txt")
    def preloadData(self):
        file = open(str(self.file_path.absolute()),"r")
        X = []
        Y = []
        lines = file.readlines()
        unique_names =[]
        for line in lines:
            line =  line.split(";")
            unique_names.append(line[0])
        unique_names = np.unique(unique_names)

        #
        for name in unique_names:
            image_path = self.folder_path.absolute().as_posix()+"/"+name
            img = cv2.imread(image_path)
            coordinates = []
            for line in lines:
                line = line.split(";")
                n = line[0]
                if n==name:
                    coordinates.append(Region(int(line[1]), int(line[2]), int(line[3]), int(line[4])))
            X.append(img)
            Y.append(coordinates)




        return X,Y
class TSDTFLabeledDataProvider:
    def __init__(self,path,output_dir = None,split=1.0):
        self.folder_path = Path(path)
        self.output_dir = Path(output_dir) if output_dir is not None else self.folder_path
        assert (split>=0.0 and split<=1.0)
        self.split = split

    def _convert_into_tf_example(self,example):
        img = example[0]
        try:
            height, width, dim = img.shape
        except Exception as e:
            print (example[1])
        path = Path(example[1])
        with tf.gfile.GFile(path.absolute().as_posix(), 'rb') as fid:
            encoded_img = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_img)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')

        key = hashlib.sha256(encoded_img).hexdigest()


        filename = path.name

        #image_format = filename.split(".")
        image_format = b'JPEG'

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        # add labels
        for label in example[2]:
            xmins.append(float(label.p1[0])/width)
            xmaxs.append(float(label.p2[0])/width)
            ymins.append(float(label.p1[1])/height)
            ymaxs.append(float(label.p2[1])/height)
            classes_text.append(label.name)
            classes.append(label.id)


        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_img),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example
    def read_examples_from_folder(self):
        folder = os.listdir(str(self.folder_path.absolute()))
        unique_names = {}
        examples = []

        for file in folder:
            if (".xml" in file):
                xml_path = self.folder_path.absolute().as_posix()+"/"+file
                xml_file = minidom.parse(xml_path)
                img_path = self.folder_path.absolute().as_posix()+"/"+ xml_file.getElementsByTagName('filename')[0].firstChild.nodeValue
                img = cv2.imread(img_path)
                xml_labels = xml_file.getElementsByTagName('object')
                if len(xml_labels)<1:
                    continue
                labels = []

                for label in xml_labels:
                    class_name = label.getElementsByTagName('name')[0].firstChild.nodeValue
                    # for id generation
                    if not str(class_name) in unique_names:
                        unique_names[str(class_name)] = len(unique_names.keys())+1

                    class_id = unique_names[str(class_name)]# must start from 1 for object detection api
                    #print str(class_id)+" "+class_name
                    x1 = int(label.getElementsByTagName('xmin')[0].firstChild.nodeValue)
                    y1 = int(label.getElementsByTagName('ymin')[0].firstChild.nodeValue)
                    x2 = int(label.getElementsByTagName('xmax')[0].firstChild.nodeValue)
                    y2 = int(label.getElementsByTagName('ymax')[0].firstChild.nodeValue)
                    xmin = min(x1,x2)
                    xmax = max(x1,x2)
                    ymin = min(y1,y2)
                    ymax = max(y1,y2)
                    l = Label(class_name,class_id,xmin,ymin,xmax,ymax)
                    labels.append(l)
                examples.append((img,img_path,labels))
        return examples,unique_names

    def create_tfrecord(self):

        output_directory = self.output_dir.as_posix() if self.output_dir is not None else self.folder_path.as_posix();
        print("set output directory to:"+str(output_directory))
        training_output_path = output_directory+"/training_set.record"
        validation_output_path = output_directory + "/validation_set.record"
        folder = os.listdir(str(self.folder_path.absolute()))
        unique_names = {}
        print("Init Tensorflow Writer...")
        training_writer = tf.python_io.TFRecordWriter(training_output_path)
        validation_writer = tf.python_io.TFRecordWriter(validation_output_path)
        print("start to write examples...")
        number_files = len(folder)/2;
        counter = 0;
        for file in folder:
            if (".xml" in file):
                xml_path = self.folder_path.absolute().as_posix() + "/" + file
                xml_file = minidom.parse(xml_path)
                img_path = self.folder_path.absolute().as_posix() + "/" + xml_file.getElementsByTagName('filename')[0].firstChild.nodeValue
                img = cv2.imread(img_path)
                xml_labels = xml_file.getElementsByTagName('object')
                if len(xml_labels) < 1:
                    continue
                labels = []

                for label in xml_labels:
                    class_name = label.getElementsByTagName('name')[0].firstChild.nodeValue
                    # for id generation
                    if not str(class_name) in unique_names:
                        unique_names[str(class_name)] = len(unique_names.keys()) + 1

                    class_id = unique_names[str(class_name)]  # must start from 1 for object detection api
                    # print str(class_id)+" "+class_name
                    x1 = int(label.getElementsByTagName('xmin')[0].firstChild.nodeValue)
                    y1 = int(label.getElementsByTagName('ymin')[0].firstChild.nodeValue)
                    x2 = int(label.getElementsByTagName('xmax')[0].firstChild.nodeValue)
                    y2 = int(label.getElementsByTagName('ymax')[0].firstChild.nodeValue)
                    l = Label(class_name, class_id, x1, y1, x2, y2)
                    labels.append(l)
                example = (img, img_path, labels)
                tf_example = self._convert_into_tf_example(example)
                # uniform distribution of training and validation data
                split_probability = np.random.random_sample()
                if split_probability <= self.split:
                    training_writer.write(tf_example.SerializeToString())
                else:
                    validation_writer.write(tf_example.SerializeToString())


        validation_writer.close()
        training_writer.close()
        # generate class_map
        print("open or create meta files...")
        data_meta_file = open(self.folder_path.absolute().as_posix() + "/data_meta.info", "w")
        label_map = open(self.folder_path.absolute().as_posix() + "/label_map.pbtxt", "w")

        print("write label map...")
        for i, key in enumerate(unique_names.keys()):
            tf_label = "item {\n \tid: " + str(unique_names[key]) + "\n \tname: \'" + str(
                key) + "\'\n}\n"  # must start from 1 for object detection api
            label_map.write(tf_label)


        label_map.close()
        #print ("create samples list")
        #for example in examples:
        #    labels = example[2]
        #    name = Path(example[1]).name
        #    for label in labels:
        #        text_line = str(name)+"\t"+str(label.p1[0])+"\t"+str(label.p1[1])+"\t"+str(label.p2[0])+"\t"+str(label.p2[1])+"\t"+str(label.name)+"\n"
        #        data_meta_file.write(text_line)
        #data_meta_file.close()
        print ("export done.")
    def create_classification_set(self,output_directory,w=40,h=40):

        assert( os.path.exists(output_directory) and os.path.isdir(output_directory), "invalid output dir")
        folder = os.listdir(str(self.folder_path.absolute()))
        unique_names = {}
        print("start to write examples...")
        number_files = len(folder)/2;
        counter = 0;
        for file in folder:
            if (".xml" in file):
                xml_path = self.folder_path.absolute().as_posix() + "/" + file
                xml_file = minidom.parse(xml_path)
                img_path = self.folder_path.absolute().as_posix() + "/" + xml_file.getElementsByTagName('filename')[0].firstChild.nodeValue
                img = cv2.imread(img_path)
                xml_labels = xml_file.getElementsByTagName('object')
                if len(xml_labels) < 1:
                    continue
                labels = []

                for label in xml_labels:
                    class_name = label.getElementsByTagName('name')[0].firstChild.nodeValue
                    # for id generation
                    if not str(class_name) in unique_names:
                        unique_names[str(class_name)] = len(unique_names.keys()) + 1

                    class_id = unique_names[str(class_name)]  # must start from 1 for object detection api
                    # print str(class_id)+" "+class_name
                    x1 = int(label.getElementsByTagName('xmin')[0].firstChild.nodeValue)
                    y1 = int(label.getElementsByTagName('ymin')[0].firstChild.nodeValue)
                    x2 = int(label.getElementsByTagName('xmax')[0].firstChild.nodeValue)
                    y2 = int(label.getElementsByTagName('ymax')[0].firstChild.nodeValue)
                    l = Label(class_name, class_id, x1, y1, x2, y2)

                    sample_path_folder = output_directory+"/"+str(class_id)

                    if not os.path.exists(sample_path_folder):
                        os.makedirs(sample_path_folder)

                    file_name = "generated_"+str(binascii.hexlify(os.urandom(16)))+".jpeg"
                    file_path = sample_path_folder+"/"+file_name
                    print("write to {}".format(file_path))
                    cv2.imwrite(file_path,img=img[y1:y2,x1:x2])



        # generate class_map
        print("open or create meta files...")
        label_map = open(output_directory + "/label_map.pbtxt", "w")

        print("write label map...")
        for i, key in enumerate(unique_names.keys()):
            tf_label = "item {\n \tid: " + str(unique_names[key]) + "\n \tname: \'" + str(
                key) + "\'\n}\n"  # must start from 1 for object detection api
            label_map.write(tf_label)


        label_map.close()
        #print ("create samples list")
        #for example in examples:
        #    labels = example[2]
        #    name = Path(example[1]).name
        #    for label in labels:
        #        text_line = str(name)+"\t"+str(label.p1[0])+"\t"+str(label.p1[1])+"\t"+str(label.p2[0])+"\t"+str(label.p2[1])+"\t"+str(label.name)+"\n"
        #        data_meta_file.write(text_line)
        #data_meta_file.close()
        print ("export done.")
