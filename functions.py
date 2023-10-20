import cv2
import csv
import math

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE_UPPER =[ 362, 382, 381, 380, 374, 373, 390,249, ]
LEFT_EYE_LOWER =[ 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_EYE_UPPER=[ 33, 7, 163, 144, 145, 153, 154, 155 ]  
RIGHT_EYE_LOWER=[133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

def draw_indecies(landmarks,original_image,list_indecies,color,new_image):
    for idx in list_indecies:
        landmark = landmarks.landmark[idx]
        height, width, _ = original_image.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(new_image, (cx, cy), 1, color, -1)

def calculate_distances(landmarks, upper_indecies ,lower_indecies):
    # height, width, _ = image.shape
    # Calculate the distance between upper and lower lip landmarks
    upper_points = [landmarks.landmark[idx] for idx in upper_indecies]
    lower_points = [landmarks.landmark[idx] for idx in lower_indecies]
    
    # Calculate the Euclidean distance between corresponding points on the upper and lower lips
    # distances = [math.dist((upper.x* width, upper.y * height), (lower.x* width, lower.y * height)) for upper, lower in zip(upper_points, lower_points)]
    distances = [math.dist((upper.x, upper.y ), (lower.x, lower.y)) for upper, lower in zip(upper_points, lower_points)]

    average_distance = sum(distances) / len(distances)
    
    return distances, average_distance

def calculate_lip_distances(landmarks):
    
    # Define the indices for the mouth landmarks (for a total of 20 landmarks)
    upper_lip_landmark_indices = [191, 183, 184, 185, 80, 42, 74, 40, 81, 41, 73, 39, 82, 38, 72, 37, 13, 12, 11, 0, 
                                312, 268, 302, 267, 311, 271, 303, 269, 310, 272, 304, 270, 415, 407, 408, 409]
    lower_lip_landmark_indices= [95, 96, 77, 146, 88, 89, 90, 91, 178, 179, 180, 181, 87, 86, 85, 84, 14, 15, 16, 17
                                    ,317, 316, 315, 314, 402, 403, 404, 405, 318, 319, 320, 321, 324, 325, 307, 375]
    # Calculate the distance between upper and lower lip landmarks
    upper_lip_points = [landmarks.landmark[idx] for idx in upper_lip_landmark_indices]
    lower_lip_points = [landmarks.landmark[idx] for idx in lower_lip_landmark_indices]
    
    # Calculate the Euclidean distance between corresponding points on the upper and lower lips
    distances = [math.dist((upper.x, upper.y), (lower.x, lower.y)) for upper, lower in zip(upper_lip_points, lower_lip_points)]
    average_distance = sum(distances) / len(distances)
    
    return distances, average_distance

def write_distance_to_csv(csv_file, frame_number, distances, average_distance, is_mouth_open):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if frame_number == 0:
            writer.writerow(['Frame', *['d' + str(i) for i in range(len(distances))], 'avg_d', 'open'])
        writer.writerow([frame_number, *distances, average_distance, is_mouth_open])

def write_time_to_csv(csv_file, frame_number, is_mouth_open,local_mouth_open_time,local_mouth_open_frames,total_mouth_open_time, total_mouth_open_frames):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if frame_number == 0:
            writer.writerow(['frame_number', 'is_mouth_open','local_mouth_open_time','local_mouth_open_frames', 'total_mouth_open_time', 'total_mouth_open_frames'])
        writer.writerow([frame_number, is_mouth_open,local_mouth_open_time,local_mouth_open_frames,total_mouth_open_time, total_mouth_open_frames])