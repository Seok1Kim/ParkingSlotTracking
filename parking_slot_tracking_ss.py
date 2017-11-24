# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:25:53 2017

@author: chulh
"""

'''
import cv2

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)
'''
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import cv2, csv
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

start_frame=[107,63,96,82,50,111,81,100]
## Set1: 107
## Set2: 63
## Set3: 96
## Set4: 82
## Set5: 50
## Set6: 111
## Set7: 81
## Set8: 100

set_num = 5
FilePath = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified/set{}'.format(set_num)
SS_FilePath = 'C:/Users/Seokwon/Desktop/Parking_slot_tracking/hyu_171121/171124/rectified/set{}/output/labeled/class_1'.format(set_num)

NumFrame = start_frame[set_num-1]

CalibPara = pickle.load(open(FilePath + '/calib.p',"rb"))
Motion = pickle.load(open(FilePath + '/motion.p',"rb"))
MeterPerPixel = CalibPara['meter_per_pixel']
XPixelMeter = MeterPerPixel[0][0]
YPixelMeter = MeterPerPixel[0][1]
VehicleCenter = CalibPara['center']
VehicleCenterRow = VehicleCenter[0][1]
VehicleCenterCol = VehicleCenter[0][0]
ItoW = CalibPara['ItoW']
WtoI = CalibPara['WtoI']

SpeedData = Motion['speed']
YawRateData = Motion['yaw_rate']

# disturbance parameter for DCM matching
x_disturb = 2
y_disturb = 2
ang_disturb = 5
            
# resize scale
resize_scale = 0.2 

global pt_list
global pt_list_tracking
pt_list = []
pt_list_tracking = []

## definitions of functions
# mouse callback function
def import_image(NumFrame):
    img = cv2.imread(FilePath + '/{:08d}'.format(NumFrame) + '.jpg')
    if img is not None:
        return img
    if img is None:
        return False
    

def draw_circle(event,x,y,flags,param):
    global pt_list
    
    if event == cv2.EVENT_FLAG_LBUTTON:
        pt_list.append((x,y))
    elif event == cv2.EVENT_FLAG_RBUTTON:
        print('points are clear')
        pt_list = []             
        
def draw_points(src):
    global pt_list
    
    for pt in pt_list:        
        cv2.circle(src,pt,2,(0,0,255),-1)
        
def draw_space(src, space, color, d):
    for i in range(len(space) - 1):
        cv2.line(src, space[i], space[i+1], color, d)
        
def estimate_space(pt1, pt2, ang):
    if pt1[0] > pt2[0]:
        tmp_pt = pt1
        pt1 = pt2
        pt2 = tmp_pt
    elif pt1[0] == pt2[0]:
        if pt1[1] > pt2[1]:
            tmp_pt = pt1
            pt1 = pt2
            pt2 = tmp_pt
            
    ang = ang / 180 * np.pi # degree > radian
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
        
    m = np.sqrt(dx*dx + dy * dy)
    
    AreaBoundary = 20
    
    # unit vector
    ux = dx/m
    uy = dy/m
    
    # rotation
    ux_rot = np.cos(ang) * ux - np.sin(ang) * uy
    uy_rot = np.sin(ang) * ux + np.cos(ang) * uy
    
    pt1 = (int(pt1[0] - AreaBoundary * ux), int(pt1[1] - AreaBoundary * uy))
    pt2 = (int(pt2[0] + AreaBoundary * ux), int(pt2[1] + AreaBoundary * uy))
    
    point1 = (int(pt1[0] - AreaBoundary * ux_rot), int(pt1[1] - AreaBoundary * uy_rot))
    point2 = (int(pt2[0] - AreaBoundary * ux_rot), int(pt2[1] - AreaBoundary * uy_rot))
    point3 = (int(pt2[0] + AreaBoundary * ux_rot), int(pt2[1] + AreaBoundary * uy_rot))
    point4 = (int(pt1[0] + AreaBoundary * ux_rot), int(pt1[1] + AreaBoundary * uy_rot))
    
    space = []
    space.append(point1)
    space.append(point2)
    space.append(point3)
    space.append(point4)
    space.append(point1)
    
    return space
    
def Image_crop(space, img, scale = 1.0):
    
    arr_space = np.array(space)
        
    point1 = (arr_space[0,0], arr_space[0,1])
    point2 = (arr_space[1,0], arr_space[1,1])
    point3 = (arr_space[2,0], arr_space[2,1])
    point4 = (arr_space[3,0], arr_space[3,1])
    
    CropXLen = int(np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2))
    CropYLen = int(np.sqrt((point3[0]-point2[0])**2 + (point3[1]-point2[1])**2))
    pt_origin = [point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], point4[0], point4[1]]
    pt_origin = np.array(pt_origin, dtype=np.float32).reshape((4,2))
    pt_transform = [0, CropYLen, CropXLen, CropYLen, CropXLen, 0, 0, 0]
    pt_transform = np.array(pt_transform, dtype=np.float32).reshape((4,2))
    Trans_mat = cv2.getPerspectiveTransform(pt_origin,pt_transform)
    img_crop = cv2.warpPerspective(img, Trans_mat, (int(140), int(40)))
    img_crop = cv2.resize(img_crop,None,fx=scale,fy=scale)
    
    return img_crop

def ImagetoWorld(ImageX, ImageY):
    
    A = ItoW[0][0] * ImageX + ItoW[0][1] * ImageY + ItoW[0][2]
    B = ItoW[1][0] * ImageX + ItoW[1][1] * ImageY + ItoW[1][2]
    C = ItoW[2][0] * ImageX + ItoW[2][1] * ImageY + ItoW[2][2]
    
    WorldPoint = [A/C, B/C]
    
    return WorldPoint

def WorldtoImage(WorldX, WorldY):
    
    A = WtoI[0][0] * WorldX + WtoI[0][1] * WorldY + WtoI[0][2]
    B = WtoI[1][0] * WorldX + WtoI[1][1] * WorldY + WtoI[1][2]
    C = WtoI[2][0] * WorldX + WtoI[2][1] * WorldY + WtoI[2][2]
    
    ImagePoint = [int(A/C), int(B/C)]
    
    return ImagePoint

def null_function(x):
    pass

def calculateDT(src, DEBUG=False):    
    src_inv = np.uint8(np.zeros((src.shape[0], src.shape[1])))            
    src_inv[src == 0] = 255
        
    sobelx_pos = cv2.Sobel(src,cv2.CV_8U,1,0,ksize=1)
    sobely_pos = cv2.Sobel(src,cv2.CV_8U,0,1,ksize=1)
    sobelx_neg = cv2.Sobel(src_inv,cv2.CV_8U,1,0,ksize=1)
    sobely_neg = cv2.Sobel(src_inv,cv2.CV_8U,0,1,ksize=1)    
    
    img_sobel_inv_xpos = np.uint8(np.zeros((src.shape[0], src.shape[1])))
    img_sobel_inv_ypos = np.uint8(np.zeros((src.shape[0], src.shape[1])))
    img_sobel_inv_xneg = np.uint8(np.zeros((src.shape[0], src.shape[1])))
    img_sobel_inv_yneg = np.uint8(np.zeros((src.shape[0], src.shape[1])))
    
    img_sobel_inv_xpos[sobelx_pos == 0] = 255
    img_sobel_inv_ypos[sobely_pos == 0] = 255
    img_sobel_inv_xneg[sobelx_neg == 0] = 255
    img_sobel_inv_yneg[sobely_neg == 0] = 255

    DT_images = []
    DT_images.append(cv2.distanceTransform(img_sobel_inv_xpos, cv2.DIST_L2, maskSize=3))
    DT_images.append(cv2.distanceTransform(img_sobel_inv_ypos, cv2.DIST_L2, maskSize=3))
    DT_images.append(cv2.distanceTransform(img_sobel_inv_xneg, cv2.DIST_L2, maskSize=3))
    DT_images.append(cv2.distanceTransform(img_sobel_inv_yneg, cv2.DIST_L2, maskSize=3))
    
    
    if DEBUG == True:
        
        Debug_DT_sobelx_pos = cv2.distanceTransform(img_sobel_inv_xpos, cv2.DIST_L2, maskSize=3)
        Debug_DT_sobely_pos = cv2.distanceTransform(img_sobel_inv_ypos, cv2.DIST_L2, maskSize=3)
        Debug_DT_sobelx_neg = cv2.distanceTransform(img_sobel_inv_xneg, cv2.DIST_L2, maskSize=3)
        Debug_DT_sobely_neg = cv2.distanceTransform(img_sobel_inv_yneg, cv2.DIST_L2, maskSize=3)
        
        Debug_DT_sobelx_pos = np.uint8(255 / Debug_DT_sobelx_pos.max() * Debug_DT_sobelx_pos)
        Debug_DT_sobely_pos = np.uint8(255 / Debug_DT_sobely_pos.max() * Debug_DT_sobely_pos)
        Debug_DT_sobelx_neg = np.uint8(255 / Debug_DT_sobelx_neg.max() * Debug_DT_sobelx_neg)
        Debug_DT_sobely_neg = np.uint8(255 / Debug_DT_sobely_neg.max() * Debug_DT_sobely_neg)
        
        cv2.imshow('DT image', Debug_DT_sobelx_pos)
        cv2.imshow('DT image1', Debug_DT_sobely_pos)
        cv2.imshow('DT image2', Debug_DT_sobelx_neg)
        cv2.imshow('DT image3', Debug_DT_sobely_neg)
        
        cv2.imshow('debug',src)
        cv2.imshow('debug_inv',src_inv)
        cv2.imshow('sobel_posx',sobelx_pos)
        cv2.imshow('sobel_posy',sobely_pos)
        cv2.imshow('sobel_negx',sobelx_neg)
        cv2.imshow('sobel_negy',sobely_neg)

    
    return DT_images

def getMovement(dt_yaw, yawrate, dt_v, v):    

    delta_yaw = dt_yaw * yawrate / 180 * np.pi  # unit: rad
    delta_travel_dist = dt_v * v # unit: meter
    
    return delta_yaw, delta_travel_dist
    

def predictPosition(src_points, delta_yaw, delta_travel, center=(0,0)):
    
    dst_points = []
    for pt in src_points:
        x = pt[0]
        y = pt[1]
        
        # Translation
        x = x + delta_travel
        
        # Rotation
        x = x - center[0]
        y = y - center[1]
        
        rot_x = np.cos(delta_yaw) * x - np.sin(delta_yaw) * y
        rot_y = np.sin(delta_yaw) * x + np.cos(delta_yaw) * y
        
        x = rot_x + center[0]
        y = rot_y + center[1]
        
        dst_points.append((x, y))
            
    return  dst_points

def dt_matching(pt_list_predicted, DT_images, x_disturb, y_disturb, ang_disturb):
    
    return

def getDisturbancePoints(src_points, x_disturbance, y_disturbance, ang_disturbance):
    
    dst_points = []
    
    tmp_point1_x = src_points[0][0] + x_disturbance
    tmp_point1_y = src_points[0][1] + y_disturbance
    tmp_point2_x = src_points[1][0] + x_disturbance
    tmp_point2_y = src_points[1][1] + y_disturbance
    
    tmp_Center_x = (tmp_point1_x + tmp_point2_x) / 2
    tmp_Center_y = (tmp_point1_y + tmp_point2_y) / 2
    
    tmp_point1_x = tmp_point1_x - tmp_Center_x
    tmp_point1_y = tmp_point1_y - tmp_Center_y
    tmp_point2_x = tmp_point2_x - tmp_Center_x
    tmp_point2_y = tmp_point2_y - tmp_Center_y
    
    tmp_Rot_point1_x = np.cos(ang_disturbance) * tmp_point1_x - np.sin(ang_disturbance) * tmp_point1_y
    tmp_Rot_point1_y = np.sin(ang_disturbance) * tmp_point1_x + np.cos(ang_disturbance) * tmp_point1_y
    tmp_Rot_point2_x = np.cos(ang_disturbance) * tmp_point2_x - np.sin(ang_disturbance) * tmp_point2_y
    tmp_Rot_point2_y = np.sin(ang_disturbance) * tmp_point2_x + np.cos(ang_disturbance) * tmp_point2_y
    
    tmp_point1_x = tmp_Rot_point1_x + tmp_Center_x
    tmp_point1_y = tmp_Rot_point1_y + tmp_Center_y
    tmp_point2_x = tmp_Rot_point2_x + tmp_Center_x
    tmp_point2_y = tmp_Rot_point2_y + tmp_Center_y

    dst_points.append((tmp_point1_x, tmp_point1_y))
    dst_points.append((tmp_point2_x, tmp_point2_y))
    
    return dst_points
    
def correctPosition(src_points, src_img, DT_templates, X_DISTURB, Y_DISTURB, ANG_DISTURB):
    
    DCM_cost = []
    corrected_points = []
    
    # generate cropping region considering the disturbances
    for x in range(-X_DISTURB,X_DISTURB+1):
        for y in range(-Y_DISTURB,Y_DISTURB+1):
            for ang in range(-ANG_DISTURB,ANG_DISTURB+1):
                ang_rad = ang / 180 * np.pi
                
                # moving points by using disturbance
                disturbance_points = getDisturbancePoints(src_points, x, y, ang_rad)                
                
                # estimate space with disturbance points
                disturbance_space = estimate_space(disturbance_points[0], disturbance_points[1], -90)
                
                # Image cropping and thresholding
                disturbance_image = Image_crop(disturbance_space, src_img, resize_scale)                
                disturbance_image[disturbance_image > 200] = 255
                disturbance_image[disturbance_image < 200] = 0
                
                # getting distance transform images
                disturbance_DT_images = calculateDT(disturbance_image)
                
                # calculating directional chamfer matching cost
                cost = 0.
                for idx_img in range(0, len(DT_templates)):
                    cost += sum(sum(np.sqrt(np.square(DT_templates[idx_img] - disturbance_DT_images[idx_img]))))
                
                total_cost = cost / len(DT_templates)
                
                DCM_cost.append(total_cost)
                corrected_points.append(disturbance_points)
       
    return corrected_points[np.argmin(DCM_cost)], min(DCM_cost)

F = np.array([[1., 0, 0, 0],[0, 1., 0, 0],[0, 0, 1., 0],[0, 0, 0, 1.]])
H = 1. * np.eye((4))
Q = 1e-5 * np.eye((4))
R = 1e-1 * np.eye((4))
P = 1 * np.eye((4))

def KalmanFilter(predicted_Points, corrected_Points, P):
    predicted_state = np.array([[predicted_Points[0][0]],[predicted_Points[0][1]],[predicted_Points[1][0]],[predicted_Points[1][1]]])
    corrected_state = np.array([[corrected_Points[0][0]],[corrected_Points[0][1]],[corrected_Points[1][0]],[corrected_Points[1][1]]])
    
    P_priori = np.dot(np.dot(F, P),F.T) + Q
    K = np.dot(np.dot(P_priori, H.T), np.linalg.inv(np.dot(np.dot(H, P_priori), H.T) + R))
    
    state_priori = np.dot(F, predicted_state)
    state_posteriori = state_priori + np.dot(K,corrected_state - np.dot(H,state_priori))
    P_posteriori = P_priori - np.dot(np.dot(K,H),P_priori)
    
    filtered_points = []
    filtered_points.append((state_posteriori[0,0],state_posteriori[1,0]))
    filtered_points.append((state_posteriori[2,0],state_posteriori[3,0]))
    
    return filtered_points, P_posteriori
    
img = import_image(NumFrame)

img_new = import_image(NumFrame + 1)

cv2.namedWindow('image (t)') 
cv2.setMouseCallback('image (t)',draw_circle)

sequential_cost = []

## main
while(1):        
      
    img_crop = None    
    cv2.circle(img,(int(VehicleCenterCol), int(VehicleCenterRow)),2,(0,255,0),-1)
    
    # initialization
    space = []
    margin = 20
    if len(pt_list) == 2:
        # initial position in image coordinate
        pt_list_tracking = pt_list
        
        # crop tracking region
        space = estimate_space(pt_list[0], pt_list[1], -90)        
        img_crop = Image_crop(space, img, resize_scale)
               
        cv2.imshow('img_crop',img_crop)
        
        # import semantic segmentation image
        img_segmentation = cv2.imread(SS_FilePath +  '/{:08d}'.format(NumFrame)  + '.jpg', 0)
        if img_segmentation is None:
            break
        
        # crop semantic segmentation image
        img_crop_ss = Image_crop(space, img_segmentation, resize_scale)
            
        # thresholding croping image
        img_crop_ss[img_crop_ss > 200] = 255
        img_crop_ss[img_crop_ss < 200] = 0
        
        # distance transform for initial tracking region
        # - input: cropped semantic segmentation (gray)
        # - output: 4 directional dt images (list)
        DT_images = calculateDT(img_crop_ss)             

    else:
        cv2.destroyWindow('img_crop')
    
    # drawing
    img_debug = np.copy(img)
    img_new_debug = np.copy(img_new)
    draw_points(img_debug)
    if len(space) != 0:
        draw_space(img_debug, space, (0,255,0), 1)
        draw_space(img_new_debug, space, (0,255,0), 1)
        
    cv2.imshow('image (t)',img_debug)    
    cv2.imshow('image (t+1)',img_new_debug)
    cv2.waitKey(20)
    
    # tracking
    while(len(pt_list_tracking) == 2):
        
        start_time = time.time()
        
        NumFrame = NumFrame + 1        
        
        # load images
        img_ss = cv2.imread(SS_FilePath +  '/{:08d}'.format(NumFrame)  + '.jpg',0) # as gray image
        img_src = import_image(NumFrame)
        img_src_debug = np.copy(img_src)
                
        if (img_src is None) or (img_ss is None):
            break       
                  
        # get motion info and calculate movement in dt
        # - dt_yaw = YawRateData[NumFrame][0] - YawRateData[NumFrame-1][0]
        # - yaw = YawRateData[NumFrame][1]
        # - dt_v = SpeedData[NumFrame][0] - SpeedData[NumFrame-1][0]
        # - v = SpeedData[NumFrame][1]
        delta_yaw, delta_travel_dist = getMovement(YawRateData[NumFrame][0] - YawRateData[NumFrame-1][0], YawRateData[NumFrame][1],
                                                   SpeedData[NumFrame][0] - SpeedData[NumFrame-1][0], SpeedData[NumFrame][1])
        
#        print('Moving heading angle: ' + str(delta_yaw))
#        print('Moving Distance: ' + str(delta_travel_dist))
        
        ############################################################################################################
        # [TODO] motion prediction only for tracking points        
        ############################################################################################################ 
        
        # motion prediction
        # input
        # - point_list
        # - delta_yaw
        # - delta_travel_pixel
        # output
        # - predicted_point_list
        
        pt_list_predicted = []
        delta_travel_pixel = delta_travel_dist / XPixelMeter        
        pt_list_predicted = predictPosition(pt_list_tracking, delta_yaw, delta_travel_pixel, (VehicleCenterCol, VehicleCenterRow))
        
        # out of boundary
        crop_boundary = estimate_space(pt_list_predicted[0], pt_list_predicted[1], -90)
        arr_crop = np.array(crop_boundary)
        
        cost = -1
        if min(arr_crop[:,0]) >= 0 and max(arr_crop[:,0]) < img.shape[1] and min(arr_crop[:,1]) >= 0 and max(arr_crop[:,1]) < img.shape[0]:
            
            # position correction
            # input
            # - pt_list_predicted
            # - img_ss
            # - DT_images
            # output
            # - pt_list_corrected
            # params
            # - x_disturb
            # - y_disturb 
            # - ang_disturb
            
            pt_list_corrected, cost = correctPosition(pt_list_predicted, img_ss, DT_images, x_disturb, y_disturb, ang_disturb)

            pt_list_corrected , P = KalmanFilter(pt_list_predicted, pt_list_corrected, P)
            
            pt_list_tracking = pt_list_corrected
            
            print("cost: {}".format(cost))
        else:            
            pt_list_tracking = pt_list_predicted
            
        print(time.time() - start_time)
        
        
        sequential_cost.append((NumFrame, cost))
        
        # drawing        
        cv2.circle(img_src_debug,(int(VehicleCenterCol), int(VehicleCenterRow)),2,(0,255,0),-1)
        cv2.circle(img_src_debug,(int(pt_list_tracking[0][0]), int(pt_list_tracking[0][1])),5,(0,0,255),-1)
        cv2.circle(img_src_debug,(int(pt_list_tracking[1][0]), int(pt_list_tracking[1][1])),5,(0,0,255),-1)
#        cv2.circle(img_src_debug,(tracking_debug_point1[0], tracking_debug_point1[1]),2,(0,0,255),-1)
#        cv2.circle(img_src_debug,(tracking_debug_point2[0], tracking_debug_point2[1]),2,(0,0,255),-1)
        
       # cv2.imshow('Next image',img_src)
        cv2.imshow('Next image_debug',img_src_debug)
        
        cv2.imwrite("./hyu_171121/171124/debug/"+str(NumFrame)+".jpg",img_src_debug)
       # cv2.imwrite("b_"+str(NumFrame)+".jpg",img_src_debug)

        cv2.waitKey(20)
            
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
sequential_cost = np.array(sequential_cost)
    
fig, ax = plt.subplots()
ax.plot(sequential_cost[:,0],sequential_cost[:,1], '-o', lw=2, alpha=0.7, mfc='orange')
    
    
    
    
cv2.destroyAllWindows()