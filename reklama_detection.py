import cv2
import torch
import numpy as np
import time
import datetime
import os
import requests
from requests.auth import HTTPBasicAuth
import yaml
import sys

# Get command line arguments
channel_name = sys.argv[1] if len(sys.argv) > 1 else None
print("Advertising tracking has been powered on for",channel_name)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

video_path = config[channel_name]['stream_url']
cam_force_address = None
reconnection_time = config["reconnection_time"]
model_name = config['common']['model_name']
database_api_url = config['common']['database_api_url']
database_username = config['common']['database_username']
database_password = config['common']['database_password']
min_time = config['common']['min_time']
frame_width = config['common']['frame_width']
frame_height = config['common']['frame_height']
threshold_time_of_detection = config['common']['threshold_time_of_detection']
company_name = config['common']['company_name']
model = torch.hub.load('ultralytics/yolov5','custom', path=model_name)

model.conf = config['model']['confidence_threshold']
model.iou = config['model']['IoU_threshold']
model.multi_label = config['model']['multiple_labels']
model.amp = config['model']['amp']
model.cuda()  # GPU

output_folder = 'videos/'+channel_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

current_directory = os.getcwd()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (frame_width, frame_height)
video_writer = False
blocking = False

def color_filter(frame):
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    # Define orange color range
    lower_green = np.array([35, 50, 50])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green
    green_hsv = np.array([120, 255, 255], dtype=np.uint8)
    # Create a mask for the orange color range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result_frame = cv2.bitwise_and(color_frame, color_frame, mask=mask)
    rgb_image = cv2.cvtColor(result_frame,  cv2.COLOR_BGR2RGB)
        
    return rgb_image

def send_to_database(data):
    response = requests.post(database_api_url, json=data, auth=HTTPBasicAuth(database_username, database_password))
    if response.status_code == 201:
        print("Data sent successfully")
        print("Response:", response.json())
    else:
        print("Failed to send data")
        print("Response:", response.text)


def connect_camera(video_path):
    print("Connecting...")
    while True:
        try:
            if cam_force_address is not None:
                requests.get(cam_force_address)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                time.sleep(reconnection_time)
                raise Exception("Could not connect to a camera: {0}".format(self.video_path))

            print("Connected to a camera: {}".format(video_path), flush=True)

            break
        except Exception as e:
            print(e)

            if blocking is False:
                break

            time.sleep(reconnection_time)
    
    return cap

cap = connect_camera(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / frame_rate  # Calculate the time delay between frames

try:
    detected_time = 0
    det_last_time = 0
    last_time = time.time()
    data_send = True
    while cap.isOpened():
        current_datetime = datetime.datetime.now()
        start_time = time.time()
        ret, color_frame = cap.read()
        if not ret:
            print("Stream stops!", flush=True)
            cap = connect_camera(video_path)
            continue

        color_frame = cv2.resize(color_frame ,(frame_width, frame_height))

        rgb_image = color_filter(color_frame)
       
        color_image = np.asanyarray(rgb_image)
        results = model(color_image)
        results.render()
        detections = results.pred[0][results.pred[0][:,4] >= model.conf]
        for det in detections: 
            x1, y1, x2, y2, conf, cls = det.tolist()
            class_name = model.names[int(cls)]
            if (x1 >= 0 and x1 < frame_width/6 and y1 >= 0 and y1 < frame_height/6 and x2 >= 0 and x2 < frame_width/4 and y2 >= 0 and y2 < frame_height/4 and class_name == company_name):        
                formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                if data_send :
                    video_url = f"video_{channel_name}_{formatted_datetime}.mp4"
                    video_filename = os.path.join(output_folder, video_url) 
                    video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, frame_size)
                video_writer.write(color_frame)
                det_elapsed = time.time() - last_time
                detected_time = detected_time + det_elapsed
                det_last_time = time.time()
                data_send = False
             
        elapsed = time.time() - det_last_time
        if (elapsed > threshold_time_of_detection and data_send == False):
            video_writer.release()
            if (int(detected_time) >= min_time):
                data = { 
                        "is_active": True,
                        "duration": int(detected_time),
                        "channel_name": channel_name,
                        "company_name": company_name,
                        "video_url": current_directory+"/"+video_url,
                        "data": current_datetime.strftime("%Y-%m-%d"),
                        "time": current_datetime.strftime("%H:%M:%S")
                    }

                send_to_database(data)
                
            else:
                if(os.path.exists(video_filename)):
                    os.remove(video_filename)
            detected_time = 0
            data_send = True
            
        last_time = time.time()

        # cv2.imshow(f'{channel_name}', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        # Add a delay to achieve the desired frame rate
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

except Exception as e:
    print("Error occured: ", e)

finally:
    cap.release()
    if video_writer:
        video_writer.release()

    cv2.destroyAllWindows()
    


