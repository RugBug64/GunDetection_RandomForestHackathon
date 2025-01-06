# GunDetection_RandomForestHackathon
Here is a project to prevent gun violence, this is for the Random Forest Hackathon.

## Inspiration

School shootings have become an increasingly prevalent issue, with the frequency of such incidents rising in recent years. In 2023 alone, gun violence resulted in over 43,000 fatalities. Despite the severity of the problem, there has been insufficient progress in addressing it. This lack of action served as a significant source of inspiration for me to develop my project.

## What it does

My program utilizes a model trained on images of firearms, enabling it to detect guns within a given scene. It displays a live camera feed with bounding boxes around detected objects, specifically identifying guns and people. Additionally, the system provides an on-screen alert when a gun is detected.

## How we built it


To develop this program, I began by recognizing the need for a system that could detect specific objects( guns, people, and police officers) in real-time from a live camera feed. To achieve this, I leveraged OpenCV for capturing and processing the live video stream. For object detection, I utilized the YOLO model, a widely used and efficient deep learning framework, to identify people and guns. I fine-tuned the YOLO model using a custom dataset to enhance its performance and accuracy in detecting these objects.

Additionally, I incorporated a Roboflow model to detect police officers within the frame. Roboflow allowed me to quickly integrate a model it trained in 22 minutes, which helped detect specific classes of objects, such as police officers, in the video feed.

The program processes the live video by analyzing each frame. Using the YOLO model, it detects and marks people and guns with bounding boxes. Meanwhile, the Roboflow model identifies police officers and adds corresponding markers. If a gun is detected, the program triggers an on-screen alert. I also implemented frame skipping to optimize performance and reduce the computational load, ensuring the system operates smoothly in real-time.

Once the frames are processed, the program overlays the detection results (bounding boxes and labels) on the live feed. Additionally, the program records the output to a video file for future reference. After refining the models and optimizing the code, the system became both fast and reliable for real-time object detection in security or monitoring applications.

## Challenges we ran into

One of the challenges I encountered was the integration of police officer detection into the program. In previous projects involving real-time object detection, I faced issues where holding up a phone with an image to the camera did not always result in the system accurately identifying the object. Instead, I had to print out an image and present it to the camera for proper recognition. However, due to the constraints of this hackathon—where a printer was not available—there is a possibility that the program may not perform as expected during a demo, despite the mAP values indicating otherwise.

Additionally, Roboflow's licensing restrictions prevent users from downloading models as .pt files, which forced me to use the Roboflow API to access the model. This introduces a limitation, as the system currently cannot run offline. Nonetheless, I am continuing to explore potential solutions to address these challenges.


## Accomplishments that we're proud of

One accomplishment I take great pride in is successfully configuring my program to accurately identify a gun, even when it is displayed on a phone screen. This marks a notable improvement, as similar attempts in previous projects faced significant challenges in achieving accurate recognition under such conditions.

## What we learned

I learned how to combine multiple models into a single object detection program. I also explored different techniques to improve model training, like greyscaling and other augmentations. Additionally, I gained insights on how to reduce CPU and GPU usage to make the program run more efficiently.

## What's next for Preventing Gun Violence

The next steps for this project involve refining the police detection feature. As previously mentioned, the effectiveness of police identification remains uncertain due to challenges with live object detection models, particularly when the subject is displayed on a phone screen. However, there is potential for further enhancement in this area. Currently, the mAP (mean Average Precision) score for detecting people and guns is 89%, while the accuracy for police detection still requires improvement.

