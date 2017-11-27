# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:34:17 2017

@author: Seokwon
"""

import cv2
import os
import pickle
from tqdm import tqdm
import csv

SpeedData = []
YawRateData = []
ImageDataTime = []
ns2s = 1000000

path = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/raw/'
EXPORT_ROOT = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified'
CALIB_PATH = './Calib/calib.p'

#calib = pickle.load(open(EXPORT_ROOT+"/"+"set8"+"/calib.p","rb"))

calib = pickle.load(open(CALIB_PATH,"rb"))
rectified = calib["rectified"]
WtoI = calib["WtoI"]
ItoW = calib["ItoW"]
center = calib["center"]
img_size = calib["img_size"]
meter_per_pixel = calib["meter_per_pixel"]


img_size = (int(calib["img_size"][0][0]), int(calib["img_size"][0][1]))

for f in range(1,9):
    folder_name = 'set{}'.format(f)
    
    #make an export directory
    export_path = os.path.join(EXPORT_ROOT, folder_name)
    if os.path.isdir(export_path) == False:
        os.mkdir(export_path)
        
    files = os.listdir(os.path.join(path, folder_name))
    for file in files:
        if file.split('.')[1] == 'rec':
            print(file)
            f = os.path.join(path + folder_name, file)
            with open(f,'r') as DataCsvFile:            
                DataReader = csv.reader(DataCsvFile, delimiter='/')
                SpeedData = []
                YawRateData = []
                for row in DataReader:
                    if len(row) != 2:
                        continue
                    
                    if row[1][16:23] == 'speed_x':
                        tmp_SpeedTime = int(row[0][3:5] + row[0][6:12]) / ns2s
                        for idx_speed in range(0,len(row[1])):
                            if row[1][idx_speed] == '=':
                                tmp_Speed = float(row[1][idx_speed+1 : len(row[1])])
            
                    elif row[1][16:23] == 'yaw_rat':
                        tmp_YawRateTime = int(row[0][3:5] + row[0][6:12]) / ns2s
                        for idx_yawrate in range(0,len(row[1])):
                            if row[1][idx_yawrate] == '=':
                                tmp_YawRate = float(row[1][idx_yawrate+1 : len(row[1])])
                                
                    else:
                        speed = (tmp_SpeedTime, tmp_Speed)
                        SpeedData.append(speed)
                        yaw_rate = (tmp_YawRateTime, tmp_YawRate)
                        YawRateData.append(yaw_rate)
                        
                print("Complete importing the whole logging data!")
    
    motion = {}
    calib = {}
    calib = {'rectified': rectified, 'WtoI': WtoI, 'ItoW': ItoW, 'center':center, 'img_size': img_size,'meter_per_pixel':meter_per_pixel}
    motion = {'speed': SpeedData, 'yaw_rate':YawRateData}
    pickle.dump( calib , open( os.path.join(EXPORT_ROOT + "/" + folder_name,"/data/calib.p"), "wb" ) )
    pickle.dump( motion , open( os.path.join(EXPORT_ROOT + '/' + folder_name,"/data/motion.p"), "wb" )) 
    
    
    for i in range(len(files)):
        file = files[i]
        if len(file.split('.')) != 2:
            continue
        if file.split('.')[1] != 'png':
            print(file + ' is skipped')
            continue
            
        image_file = os.path.join(path, folder_name, file)
        image = cv2.imread(image_file)
        
        # crop the top view area
        img_top_view= image[20:459, 13:222, :]
        
        # rectify the top view image
        img_recti = cv2.warpPerspective(img_top_view, rectified, dsize=img_size)
        img_recti = cv2.cvtColor(cv2.flip(cv2.transpose(img_recti),0), cv2.COLOR_BGR2RGB)
        
        # save the rectified image
        save_file = '/images/{:08d}.jpg'.format(int(file.split('_')[-1].split('.')[0]))
        cv2.imwrite(os.path.join(export_path,save_file), img_recti)
