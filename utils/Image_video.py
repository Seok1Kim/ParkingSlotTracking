# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:39:53 2017

@author: Seokwon
"""

import cv2

folder = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified/set3/debug'

for frame in range(97, 538):
    
    tracking_image = cv2.imread(folder + '/' + str(frame) + '.jpg')
        
    cv2.imshow('tracking_image', tracking_image)
    
    
    cv2.waitKey(20)
    
folder = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified/set1/debug'

for frame in range(108, 601):
    
    tracking_image = cv2.imread(folder + '/' + str(frame) + '.jpg')
        
    cv2.imshow('tracking_image', tracking_image)
    
    
    cv2.waitKey(20)
    
folder = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified/set7/debug'

for frame in range(82, 580):
    
    tracking_image = cv2.imread(folder + '/' + str(frame) + '.jpg')
        
    cv2.imshow('tracking_image', tracking_image)
    
    
    cv2.waitKey(20)