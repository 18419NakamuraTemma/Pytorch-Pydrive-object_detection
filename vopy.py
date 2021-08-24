from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
import math


cp = 0


x1=[]
y1=[]
w1=[]
h1=[]
x2=[]
y2=[]
w2=[]
h2=[]
z1=[]
c3=0
c4=0
    
xx2=0
yy2=0
xx3=0
yy3=0
xx4=0
yy4=0

cx1=0
cy1=0
cx1_1=0
cy1_1=0
cx2=0
cy2=0
cx3=[]
    
y22=[]
x33=0
y33=0
x44=0
y44=0
ct=6



def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (255,255,255)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    inds = 10

    im= img
    for i in range(inds):
        
        
        
        x=c1[0]
        y=c1[1]
        h=c2[0]-c1[0]
        w=c2[1]-c1[1]
        

        cv2.circle(im, (int((x+h)/2), int((y+w)/2)), 10, color=(0, 0, 255), thickness=-1)
        global cp
    
        if cp!=0:
            ccp=0
            cut=0
            kyori=1000
            for j in range(cp):

              if math.sqrt( ((x+h)/2-(x2[j][len(x2[j])-1]+h2[j][len(h2[j])-1])/2.0)**2 + ((y+w)/2-(y2[j][len(y2[j])-1]+w2[j][len(w2[j])-1])/2.0)**2 )<=40 and ct-z1[j][len(z1[j])-1]<=120:
                 cut=cut+1
                 if kyori > np.sqrt( ((x+h)/2-(x2[j][len(x2[j])-1]+h2[j][len(h2[j])-1])/2.0)**2 + ((y+w)/2-(y2[j][len(y2[j])-1]+w2[j][len(w2[j])-1])/2.0)**2 ):
                     kyori=np.sqrt( ((x+h)/2-(x2[j][len(x2[j])-1]+h2[j][len(h2[j])-1])/2.0)**2 + ((y+w)/2-(y2[j][len(y2[j])-1]+w2[j][len(w2[j])-1])/2.0)**2 )
                     ccp=j
            if cut!=0:
               if ct-z1[ccp][len(z1[ccp])-1]<=120:
                     x1[ccp].append((x+h)/2.0)
                     y1[ccp].append((y+w)/2.0)


                     x2[ccp].append(c1[0])
                     y2[ccp].append(c1[1])
                     h2[ccp].append(c2[0])
                     w2[ccp].append(c2[1])    

                     z1[ccp].append(ct)
                     ccp=0
                     cut=0
               else:
                    x1.append([])
                    y1.append([])

                    x2.append([])
                    y2.append([])
                    h2.append([])
                    w2.append([])

                    z1.append([])
                    if (x+h)/2.0<1499:#1499 999
                         x1[cp].append((x+h)/2.0+cx1)
                         y1[cp].append((y+w)/2.0+cy1)
                    else:
                         x1[cp].append((x+h)/2.0+cx1_1)
                         y1[cp].append((y+w)/2.0+cy1_1)  


                    x2[cp].append(c1[0])
                    y2[cp].append(c1[1])
                    h2[cp].append(c2[0])
                    w2[cp].append(c2[1])    
                    cx3.append(cx1)
                    z1[cp].append(ct)
                    cp=cp+1
                            
            else:
                x1.append([])
                y1.append([])

                x2.append([])
                y2.append([])
                h2.append([])
                w2.append([])

                z1.append([])

                x1[cp].append((x+h)/2.0)
                y1[cp].append((y+w)/2.0)  

                x2[cp].append(c1[0])
                y2[cp].append(c1[1])
                h2[cp].append(c2[0])
                w2[cp].append(c2[1])    
                cx3.append(cx1)
                z1[cp].append(ct)
                cp=cp+1
        else:
            x1.append([])
            y1.append([])

            x1[cp].append((x+h)/2.0+cx1_1)
            y1[cp].append((y+w)/2.0+cy1_1)  


            x2.append([])
            y2.append([])
            h2.append([])
            w2.append([])

            z1.append([])

            x2[cp].append(c1[0])
            y2[cp].append(c1[1])
            h2[cp].append(c2[0])
            w2[cp].append(c2[1])    
            cx3.append(cx1)
            z1[cp].append(ct)
            cp=cp+1

                
        for i in range(cp):
            if len(x1[i])>=1 and ct-z1[i][len(z1[i])-1]<=120:

                if ct==z1[i][len(z1[i])-1]:
                    
                    xxx=x1[i]
                    yyy=y1[i]
                    xxx=np.array(xxx).astype(float)
                    yyy=np.array(yyy).astype(float)

                    
                    a,b= np.polyfit(xxx, yyy, 1)

                    x55=xxx[len(xxx)-1]+xxx[len(xxx)-1]-(xxx[0])
                            

                    y55=a * x55 + b
                    cv2.arrowedLine(im,(int(xxx[len(xxx)-1])-cx1_1,int(yyy[len(yyy)-1])-cy1_1),(int(x55-cx1_1),int(y55-cy1_1)), color=(255, 153, 153), thickness=10)
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "front2.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/boat2.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "boat.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            

            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            

            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes('data/boat.name')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
    

    
    

