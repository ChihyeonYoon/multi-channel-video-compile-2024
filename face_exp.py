import cv2

import torch
from torch import nn
from torchvision.models import swin_b, Swin_B_Weights
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn


import cv2
import mediapipe
# import face_recognition
# from imutils import face_utils


cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# detector = dlib.get_frontal_face_detector()
# predictor = face_recognition.api.pose_predictor_68_point

face_detection_ = mediapipe.solutions.face_detection
drawing = mediapipe.solutions.drawing_utils

def mediapipe_inference(frame, face_detection):
    # with face_detection_.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.detections:
        # print('\tmediapipe inference failed')
        return None
    else:
        # print('\tmediapipe inference done')

        for i, detection in enumerate(results.detections):
            x1 = int(round(detection.location_data.relative_bounding_box.xmin*frame.shape[1]))
            y1 = int(round(detection.location_data.relative_bounding_box.ymin*frame.shape[0]))
            x2 = int(round((detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width)*frame.shape[1]))
            y2 = int(round((detection.location_data.relative_bounding_box.ymin+detection.location_data.relative_bounding_box.height)*frame.shape[0]))
            
            
            return [x1, x2, y1, y2]
        
# def dlib_inference(frame):
#     org_image = frame
#     image = org_image.copy()
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     rects = detector(gray, 1)
#     lip_points = []
    
#     if len(rects) != 0:
        
#         for (i, rect) in enumerate(rects):
#             if i > 0:
#                 break
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)

#             # print('\tdlib inference done')
#             x1 = shape.min(axis=0)[0]
#             x2 = shape.max(axis=0)[0]
#             y1 = shape.min(axis=0)[1]
#             y2 = shape.max(axis=0)[1]
            
        
#         return [x1, x2, y1, y2]
#     else:
#         # print('\tdlib inference failed')
#         return None

# def get_rectsize(x1,y1,x2,y2):
#     w = abs(x2 - x1) + 1
#     h = abs(y2 - y1) + 1
#     cx = x1 + w // 2
#     cy = y1 + h // 2

#     return w*h, [cx, cy]

def get_rectsize(x1, x2, y1, y2):
    w = abs(x2 - x1) + 1
    h = abs(y2 - y1) + 1
    cx = x1 + w // 2
    cy = y1 + h // 2

    return w*h, [cx, cy]

class swin_binary_module(nn.Module):
    def __init__(self):
        super(swin_binary_module, self).__init__()
        self.Lin_1 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.Lin_1(x)
        x = self.softmax(x)

        return x
    
class swin_face(nn.Module):
    def __init__(self, pretrained=False):
        super(swin_face, self).__init__()
        # self.model = swin_b()
        if pretrained:
            self.model = swin_b(weights = Swin_B_Weights.IMAGENET1K_V1)
            self.model.head = swin_binary_module()
            
        if not pretrained:
            self.model = swin_b()
            self.model.head = swin_binary_module()
        self.loss = CrossEntropyLoss().cuda()

    def forward(self, x):
        return self.model(x)
    