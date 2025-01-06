import cv2
from ultralytics import YOLO
import numpy as np
from roboflow import Roboflow


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  

output_file = 'output_video.avi'  

fps = 7.0 

frame_size = (640, 480)  

out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)


yolo_model = YOLO('/Users/raghuveer/Desktop/New Folder With Items 5/Gun Detection Project/best.pt')


rf = Roboflow(api_key="######")  #I commented out my API KEY, replace with your own API KEY
roboflow_project = rf.workspace("security-training").project("police-jq67t")


roboflow_model = roboflow_project.version(2).model  

while True:
 
    ret, frame = cap.read()
    if not ret:
        break

  
    yolo_results = yolo_model(frame)
    
    people = []
    guns = []

    try:
        
        for result in yolo_results:
            boxes = result.boxes

        for box in boxes:
            cls = np.array(box.cls).astype(int)
            box_ = np.array(box.xywh).astype(int)
            if cls[0] == 1:  
                people.append(box_[0])
            else: 
                guns.append(box_[0])

        
        rf_results = roboflow_model.predict(frame, confidence=5)  

        police = []
        for prediction in rf_results.json()['predictions']:
            if prediction['class'] == 'police':  
               
                bbox = prediction['bbox']
                police.append(bbox)

       
        for item in people:
            x, y, x_max, y_max = item
            x = int(x - x_max / 2)
            y = int(y - y_max / 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + x_max), int(y + y_max)), (0, 255, 0), 2)
            cv2.putText(frame, "person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for item in guns:
            x, y, x_max, y_max = item
            x = int(x - x_max / 2)
            y = int(y - y_max / 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + x_max), int(y + y_max)), (255, 0, 0), 2)
            cv2.putText(frame, "gun", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        
        for bbox in police:
            x, y, w, h = bbox  
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow for police
            cv2.putText(frame, "Police", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        
        if guns:
            cv2.putText(frame, "Gun Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

       
        cv2.imshow('Gun Detection', frame)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        cv2.imshow('Gun Detection', frame)

    out.write(frame)

   
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
out.release()
