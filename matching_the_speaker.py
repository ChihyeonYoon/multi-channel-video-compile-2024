from ast import List
import cv2
import mediapipe
import numpy as np
import traceback

def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

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

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

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
                return lip_coords, state # coords: x1, y1, x2, y2, state: True or False
            else:
                return None, None # coords: None, state: None

        except Exception as e:
            print(e)
            traceback.print_exc()
            return None, None # coords: None, state: None
        

def segment_speaker_match(segment:tuple, video_list:List):
    start_frame, end_frame = segment
    
