import os
import cv2
import numpy as np
import math
import multiprocessing as mp
import time
import mediapipe
import traceback
import random
import torch
from PIL import Image
from argparse import ArgumentParser
import json    
import logging

import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

def get_model(model_name, num_classes):
    if model_name == 'swin_v2_b':
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        model = swin_v2_b(weights=weights)
        preprocess = weights.transforms()
        
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model, preprocess

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

# 회전 행렬 함수
def create_rotation_matrix(yaw, pitch, roll):
    # Yaw (좌우 회전)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Pitch (상하 회전)
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Roll (기울기 회전)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 최종 회전 행렬
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    return R

# 얼굴의 기울기를 보정하는 함수 (yaw, pitch, roll)
def correct_angle(landmarks, point, left_eye_index, right_eye_index):
    left_eye = np.array(landmarks[left_eye_index])
    right_eye = np.array(landmarks[right_eye_index])
    point = np.array(point)
    
    eye_center = (left_eye + right_eye) / 2.0
    
    yaw = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    
    pitch = np.arctan2(right_eye[2] - left_eye[2], right_eye[0] - left_eye[0])
    
    roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[2] - left_eye[2])
    
    # 3D 회전 행렬 생성
    rotation_matrix = create_rotation_matrix(yaw, pitch, roll)
    
    # 포인트 보정
    corrected_point = np.dot(rotation_matrix, point - eye_center) + eye_center
    return corrected_point

def is_mouth_open(landmarks, threshold=0.05):
    upper_lip_top_index = 13 
    lower_lip_bottom_index = 14 
    mouth_left_corner_index = 61 
    mouth_right_corner_index = 291  
    left_eye_index = 130 # 130
    right_eye_index = 263 # 263
    
    upper_lip_top = landmarks[upper_lip_top_index]
    lower_lip_bottom = landmarks[lower_lip_bottom_index]
    mouth_left_corner = landmarks[mouth_left_corner_index]
    mouth_right_corner = landmarks[mouth_right_corner_index]

    mouth_height = euclidean_distance(upper_lip_top, lower_lip_bottom)
    
    mouth_width = euclidean_distance(mouth_left_corner, mouth_right_corner)
    
    ratio = mouth_height / mouth_width

    return ratio > threshold

def lip_detection_in_video(video_path, early_q, total_frames):
    # from mediapipe.python.solution import face_mesh as mp_face_mesh
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    # print(face_mesh)
    
    def lip_detection_in_frame(frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lip_indexes = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
                    321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
                    269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
                    14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81,
                    81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308,]

        lip_contour = [61, 185, 40, 39, 37, 0, 267, 270] + [146, 91, 181, 84, 17, 314, 405, 321] # lip_high: 0, lip_low: 17
        
        upper_lip_top_index = 13 
        lower_lip_bottom_index = 14 
        mouth_left_corner_index = 61 
        mouth_right_corner_index = 291  
        
        lip_landmarks = []
        lip_coords = []
        state = None
        try:
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                
                face_landmarks = results.multi_face_landmarks[0]
                # print(len(face_landmarks.landmark))
                landmarks = {i: [landmark.x, landmark.y, landmark.z] for i, landmark in enumerate(face_landmarks.landmark)}
                
                state = is_mouth_open(landmarks)
                lip_landmarks = [landmarks[i][:-1] for i in lip_contour]
                lip_landmarks = [[round(x*frame.shape[1]), round(y*frame.shape[0])] for x, y in lip_landmarks]
                # print(lip_landmarks)
            
                # lip_coords = list(cv2.boundingRect(np.array(lip_landmarks))) # x, y, w, h
                x,y,w,h = cv2.boundingRect(np.array(lip_landmarks))
                # lip_coords = [lip_coords[0], lip_coords[1], lip_coords[0]+lip_coords[2], lip_coords[1]+lip_coords[3]] # x1, y1, x2, y2
                lip_coords = [x, y, x+w, y+h]
                # print(lip_coords)
                if all(x > 0 for x in lip_coords):
                    return lip_coords, state # coords: x1, y1, x2, y2, state: True or False
                else:
                    return None, None
            else:
                return None, None # coords: None, state: None

        except Exception as e:
            print(e)
            traceback.print_exc()
            return None, None # coords: None, state: None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    current_frame = 0
    faild_frames =[]
    last_rect, last_state = None, None
    tmp_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'video: {video_path.split("/")[-1]} min_total_frames: {tmp_total_frames}')
    # exit()

    print(f"{mp.current_process().name} Processing video: {video_path}")
    while(cap.isOpened() and current_frame <= total_frames):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if ret:
            rect, state = lip_detection_in_frame(frame) # rect: x1, y1, x2, y2, state: True or False

            if rect is not None: # rect is not None and state is not None
                last_rect, last_state = rect, state
                lip_roi = frame[rect[1]:rect[3], rect[0]:rect[2]]
                early_q.put((current_frame, lip_roi, state)) # current_frame, lip_roi, state
                last_lip_roi = lip_roi
                
            else: # rect is None or state is None
                if last_rect:
                    # lip_roi = frame[last_rect[1]:last_rect[3], last_rect[0]:last_rect[2]]
                    early_q.put((current_frame, last_lip_roi, last_state)) # current_frame, lip_roi, state
                faild_frames.append(current_frame)
        else: # 
            print(f"frame {current_frame}: No frame")
            early_q.put((current_frame, None, None)) # current_frame, lip_roi, state
        
        if current_frame >= total_frames:
            break
        # break
        
    # print(f"lip_roi: {lip_roi}, state: {state}")
    print(f'{video_path.split("/")[-1]}: Done, failed frames: {len(faild_frames)}')
    cap.release()

def infer_lip_state(early_q, result_list, model_name, weights):
    # early_q item: (frame_number, lip_roi, state) or (frame_number, None, None) or 'LAST'

    fix_seed(999)
    model, preprocess = get_model(model_name=model_name, num_classes=2)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    print(f"{mp.current_process().name} Model loaded")

    frame_number_batch = []  # keep len 30
    img_batch = []  # keep len 30
    state_batch = []  # keep len 30

    while True:
        if not early_q.empty():
            try:
                item = early_q.get(block=False) # item: (frame_number, lip_roi, state) or (frame_number, None, None) or 'LAST'
                if item == 'LAST': 
                    # process remaining items
                    print(f"{mp.current_process().name} Last item received")
                    if len(frame_number_batch) > 0:
                        img_batch = [preprocess(img) for img in img_batch] 
                        img_batch = torch.stack(img_batch).cuda()
                        with torch.no_grad():
                            outputs = model(img_batch)
                            outputs = nn.Softmax(dim=1)(outputs)
                            outputs = outputs.cpu().numpy().tolist()
                            # outputs = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                            for frame_number, state, output in zip(frame_number_batch, state_batch, outputs):
                                result_list[frame_number-1] = (state, output)
                            frame_number_batch = []
                            img_batch = []
                            state_batch = []
                    break
                
                
                elif item[1] is not None: # lip_roi is not None
                    """
                    gather 30 frames, then process if full
                    when processing, change the items to inferenced value of each index in result_list
                    """

                    frame_number_batch.append(item[0])
                    img_batch.append(Image.fromarray(item[1],'RGB')) # lip_roi:
                    state_batch.append(item[2])

                    if len(frame_number_batch) == 30:
                        print(f"{mp.current_process().name} Processing frames: {frame_number_batch[0]} ~ {frame_number_batch[-1]}")
                        img_batch = [preprocess(img) for img in img_batch] # zero_division error occurs
                        img_batch = torch.stack(img_batch).cuda()
                        with torch.no_grad():
                            outputs = model(img_batch)
                            outputs = nn.Softmax(dim=1)(outputs)
                            outputs = outputs.cpu().numpy().tolist()
                            # outputs = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                            for frame_number, state, output in zip(frame_number_batch, state_batch, outputs):
                                result_list[frame_number-1] = (state, output)
                            frame_number_batch = []
                            img_batch = []
                            state_batch = []

                elif item[1] is None: # lip_roi is None

                    print(f"{mp.current_process().name} Frame {item[0]-1}: No lip_roi")
                    result_list[item[0]-1] = (None, None)
                    pass


            except Exception as e:
                print(e)
                traceback.print_exc()
                continue