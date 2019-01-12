import os
import cv2
import numpy as np
import time
from shape import Shape
from pascal_voc_io import PascalVocWriter
from pathlib import Path
import os
import binascii
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from DataProvider import TSDTFLabeledDataProvider

output_dir = Path('output')
samples_dir = Path('objects')
bg_dir = Path('backgrounds')
max_yaw = 80
max_rot =15
# scale factor in percents 0.1 => 10% of original size
min_object_scale = 0.125
max_object_scale = 0.8
# data distribution settings
max_samples_per_image = 4
number_sample_images = 70000
number_classes = 78;
max_samples_per_class = (max_samples_per_image/2)*(number_sample_images/number_classes)
output_image_size_wh = (600, 600)
# define convolution kernels
filter_kernels=[]
b_k_size = 3
b_k_avg_size = 6
kernel_avg = np.ones((b_k_size, b_k_size), np.float32) / b_k_avg_size
mb_k_size = 3
kernel_motion_blur = np.zeros((mb_k_size, mb_k_size))
kernel_motion_blur[int((mb_k_size-1)/2), :] = np.ones(mb_k_size)
kernel_motion_blur = kernel_motion_blur / mb_k_size
filter_kernels.append(kernel_avg)
filter_kernels.append(kernel_motion_blur)
# set max number of filter/convolution operations
max_number_filter_apps_on_samples = 0
max_number_filter_apps_on_background = 1
# object transparancy (sum of the values bellow must be v<=1)
min_object_transparancy = 0.6
max_object_transparancy = 0.4
##
min_sample_size = 30 #px

generate_tf = True
tf_split = 0.8

def transparentOverlay(src, overlay, pos=(0, 0), scale=1,alpha_v=1.0):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255)*alpha_v  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


## load samples and backgrounds
sample_paths = [(samples_dir.as_posix()+"/"+ name,name.replace(".png","")) for name in  os.listdir(samples_dir.as_posix())];
samples = [(cv2.imread(sample_path[0], cv2.IMREAD_UNCHANGED),sample_path[1]) for sample_path in sample_paths]
bg_paths = [bg_dir.as_posix()+"/"+ name for name in  os.listdir(bg_dir.as_posix())];
backgrounds = [cv2.imread(bg_path, cv2.IMREAD_UNCHANGED) for bg_path in bg_paths]
counter = 0;
## auxiliaries
samples_map = {} # for sample distribution stats
while counter < number_sample_images:
    try:
        print(str(counter) +" of " + str(number_sample_images) + " samples generated..")
        background = backgrounds[np.random.random_integers(0,len(backgrounds)-1)]
        background = cv2.resize(background,output_image_size_wh,cv2.INTER_CUBIC)
        image_size = background.shape
        file_path = output_dir.as_posix()
        file_name = "generated_"+str(binascii.hexlify(os.urandom(16)))
        xml_file_name = file_name+".xml"
        image_file_name = file_name+".jpeg"
        xml_file_path = file_path+"/"+xml_file_name
        image_file_path = file_path+"/"+image_file_name
        labelWriter = PascalVocWriter(foldername=output_dir.name,imgSize=image_size,filename=image_file_name,localImgPath=image_file_path)
        res = background.copy()
        shapes = []
        num_samples_per_img = np.random.random_integers(1,max_samples_per_image)
        while len(shapes)<num_samples_per_img:
            ## transform sample
            sample_index = np.random.random_integers(0,len(samples)-1);
            object_sample = samples[sample_index][0].copy();
            label_name = samples[sample_index][1];
            label_name = label_name.split('.')[0];
            if label_name not in samples_map.keys():
                samples_map[label_name]=0;
                print ("New Stats for {}".format(label_name))
            if samples_map[label_name]>max_samples_per_class:
                print ("Maximum reached for {}".format(label_name))
                continue
            if (len(samples_map.keys())>=number_classes and np.min(samples_map.values())>=max_samples_per_class):

                print("maximum samples per class condition reached. Stop Generation.")
                counter = number_sample_images
                break
            h, w, ch = object_sample.shape
            # transform perspectivly
            yaw_dir = np.random.random_integers(0, 1) # 1 is right 0 is left
            yaw = np.random.random_integers(0, max_yaw)
            pts1 = np.float32([[0, 0], [w, 0],
                               [0, h], [w, h]])
            yaw_left = (1-yaw_dir)*yaw
            yaw_right = yaw_dir*yaw
            pts2 = np.float32([[yaw_left, yaw_left / 6], [w - yaw_right, (yaw_right / 6)],
                               [yaw_left, h - (yaw_left / 6)], [w - yaw_right, h - (yaw_right / 6)]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            object_sample = cv2.warpPerspective(object_sample, M, (h, w))
            ## rotate
            center = (w / 2, h / 2)
            rotation = np.random.random_integers(-max_rot, max_rot)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            object_sample = cv2.warpAffine(object_sample, M, (w, h))

            ## rescale
            resize_factor = max(min_object_scale,max_object_scale*np.random.random())
            object_sample = cv2.resize(object_sample,None,fx=resize_factor,fy=resize_factor,interpolation=cv2.INTER_CUBIC)
            ## find min bounding box by dilating and finding contours of the sample
            h_bg,w_bg,chann_bg = background.shape
            gray_obj = cv2.cvtColor(object_sample, cv2.COLOR_RGBA2GRAY);
            ret, gray_obj = cv2.threshold(gray_obj, 1, 255, cv2.THRESH_BINARY)

            gray_obj = cv2.morphologyEx(gray_obj, cv2.MORPH_DILATE,(5,5),iterations=3)

            _, conts, _ = cv2.findContours(gray_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                (bx, by, bw, bh) = cv2.boundingRect(conts[0])
                if bw<min_sample_size or bh<min_sample_size:
                    print("sample is too small. Skip it...")
                    continue
            except:
                #print("Error while trying to find a boundig box")
                continue
            x1 = int(max(0,(np.random.random()*(w_bg-bw))));
            y1 = int(max(0,(np.random.random()*(h_bg-bh))));
            x2 = int(min(w_bg,x1+bw));
            y2 = int(min(h_bg,y1+bh));

            ## apply filters
            count_filter_apps = 0
            number_filters = np.random.random_integers(0, max_number_filter_apps_on_samples)
            while count_filter_apps < number_filters:
                kernel = filter_kernels[np.random.random_integers(0, len(filter_kernels) - 1)]
                object_sample = cv2.filter2D(object_sample, -1, kernel)
                count_filter_apps = count_filter_apps + 1


            pos = (x1-bx,y1-by)
            shape = Shape(label=label_name)
            shape.addPoint(QPointF(x1,y1))
            shape.addPoint(QPointF(x2,y1))
            shape.addPoint(QPointF(x1,y2))
            shape.addPoint(QPointF(x2,y2))
            random_alpha = np.random.random()*max_object_transparancy

            #print("look for a place for new shape")
            is_valid_shape = True;
            for s in shapes:
                if shape.intersects(s):
                    is_valid_shape=False;
                    break

            if is_valid_shape:
                #print ("add shape for label:" + label_name)
                shapes.append(shape)
                res = transparentOverlay(res, object_sample, pos=pos, scale=1, alpha_v=min_object_transparancy+random_alpha)
                labelWriter.addBndBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2, name=label_name, difficult=0)
                shape = None
                samples_map[label_name] = samples_map[label_name] + 1
                continue
            else:
                #print("overlapping "+str(shape))
                shape=None
                break

            #print("shapes: "+str(len(shapes)))
        ## apply convolutioons once on result image
        number_filters = np.random.random_integers(0, max_number_filter_apps_on_background)
        count_filter_apps = 0
        while count_filter_apps < number_filters:
            kernel = filter_kernels[np.random.random_integers(0, len(filter_kernels) - 1)]
            res = cv2.filter2D(res, -1, kernel)
            count_filter_apps = count_filter_apps + 1

        ##
        ## color manipulation on entire result
        h,s,v = cv2.split(cv2.cvtColor(res,cv2.COLOR_RGB2HSV))
        random_hue = np.random.random_integers(-10,10)
        h_min = random_hue+np.min(h)
        h_max = random_hue+np.max(h)
        if h_min<0 or h_max>255:
            random_hue=0
        h = (h+random_hue)
        s/=np.random.random_integers(1,2);
        v/=np.random.random_integers(1,2);
        res = cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2RGB)
        ##
        cv2.imwrite(image_file_path,res)
        labelWriter.save(targetFile=xml_file_path)
        counter = counter + 1;
        #print ("COUNTER: "+str(counter))
    except Exception as e:
        print ("Something went wrong: {} - continue anyway...".format(e))

## done - delete variables
sample_paths = None
samples = None
bg_paths = None
backgrounds = None
## generate tf record if selected
if generate_tf:
    tsd = TSDTFLabeledDataProvider(output_dir, split=tf_split)
    tsd.create_tfrecord()
