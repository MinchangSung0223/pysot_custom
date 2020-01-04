from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame
print("Class Number ( ex: 0,1,2 ): ")
class_number = input()

print("Threshold ( ex: 1,1.5,1 ): ")
m_threshold = input()
MASK_THRESHOLD = 0.1465*np.array(m_threshold.split(','),dtype=float)
classnumber = np.array(class_number.split(','),dtype=int)
class_len = len(classnumber)
mask_len = len(MASK_THRESHOLD)
if not(class_len==mask_len) or classnumber.max() > 14 :
   print("-------------INPUT ERROR------------")
   exit()
print("Input Class Number : ",classnumber)

print("Input MASK THRESHOLD : ",MASK_THRESHOLD)
print("Image Name : ")
imagename = input()
print("Image Name : ",imagename)
model = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tracker = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def main():
    # load config
    cfg.merge_from_file('config.yaml')
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    for mi in range(0,class_len):
        model[mi] = ModelBuilder()
        # load model
        model[mi].load_state_dict(torch.load('model.pth',
        map_location=lambda storage, loc: storage.cpu()))
        model[mi].eval().to(device)
        # build tracker
        tracker[mi] = build_tracker(model[mi])

    
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    n = 0
    for frame in get_frames(args.video_name):
        n= n+1
        width = frame.shape[1]
        height = frame.shape[0]
        print(class_len)
        if first_frame:
            for i in range(0,class_len):
               try:
                   init_rect = cv2.selectROI(video_name, frame, False, False)
               except:
                   exit()
               tracker[i].init(frame, init_rect)
            first_frame = False
        else:
          frame_temp = frame.copy()
          txt_data =""
          for i in range(0,class_len):
            outputs = tracker[i].track(frame_temp)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                bbox = list(map(int, outputs['bbox']))
                x= bbox[0]
                y= bbox[1]
                w= bbox[2]
                h= bbox[3]
                                
                x_ = float(x+w/2)/width
                y_ = float(y+h/2)/height
                w_ = float(w)/width
                h_ = float(h)/height
                txt_data = txt_data+str(classnumber[i])+" "+ str(x_)+" "+ str(y_)+" "+ str(w_)+" "+ str(h_)+"\n"
                
                cv2.polylines(frame_temp, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                cv2.rectangle(frame_temp, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (255, 0, 0), 3)
                mask = ((outputs['mask'] > MASK_THRESHOLD[i]) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                #frame_temp = cv2.addWeighted(frame_temp, 0.77, mask, 0.23, -1)
            else:
                print("dddddddddddddd2")
                bbox = list(map(int, outputs['bbox']))
                x= bbox[0]
                y= bbox[1]
                w= bbox[2]
                h= bbox[3]
                                
                x_ = float(x+w/2)/width
                y_ = float(y+h/2)/height
                w_ = float(w)/width
                h_ = float(h)/height

                txt_data = txt_data+str(classnumber[i])+" "+ str(x_)+" "+ str(y_)+" "+ str(w_)+" "+ str(h_)+"\n"

                frame_temp = cv2.rectangle(frame_temp, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
          filename = imagename+str(n)+".png"
          cv2.imwrite(filename,frame)
          txt_filename = imagename+str(n)+".txt"
          try:
             print(txt_data)
             f = open(txt_filename, 'w')   
             f.write(txt_data)
          except:
             f.close()
          cv2.imshow(video_name, frame_temp)
          cv2.waitKey(40)


if __name__ == '__main__':
    main()
