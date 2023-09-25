import cv2
import mediapipe as mp
import serial
import time
import imutils


from mediapipe.framework.formats import location_data_pb2
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# arduino = serial.Serial(port='COM10', baudrate=9600, timeout=.1)
# def write_read(x):
#     arduino.write(bytes(x,'utf-8'))
#     time.sleep(0.1)


# write_read("0 0")
# 'http://192.168.137.132:8080/video'
# For webcam input:
i=0

cap = cv2.VideoCapture('http://192.168.63.71:4747/video') #http://172.17.7.17:8080/video
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
  while True:
      
    while cap.isOpened():
        
      success, image = cap.read()
      image = imutils.resize(image, width=1000)
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)
      # print(results.detections)
      (h, w) = image.shape[:2] #w:image-width and h:image-height
      cv2.circle(image, (w//2, h//2), 7, (255, 255, 255), -1) 

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
      # cv2.line(image,0,0,"red", 0.5)   

      

      
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
          
          location_data = detection.location_data
          if location_data.format == location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
              bb = location_data.relative_bounding_box
              bb_box = [
                  bb.xmin, bb.ymin,
                  bb.width, bb.height,
              ]
              
              # Assuming you have the relative bounding box coordinates
              x, y, width, height = bb_box[0], bb_box[1], bb_box[2], bb_box[3]

              # Calculate the center point
              center_x = (x + width / 2)*w
              center_y = (y + height / 2)*h
              center = (int(center_x), int(center_y))
              cv2.circle(image, (int(center_x), int(center_y)), 7, (255, 255, 255), -1) 

              # Till Line 85 This saves a cropped face from the detected faces
              x1 = int(x*w)
              x2 = int((x+width)*w)
              y1 = int(y*h)
              y2 = int((y+height)*h)

              print(x1,x2,y1,y2)

              cropped_image = image[y1-20:y2, x1-20:x2]
              cv2.imwrite(f"image/cropped_image{i}.jpg", cropped_image)


              cv2.line(image, (w//2,h//2), center, (0, 255, 0), 2)
              deg_x = int(((w/2 - center[0])/(w/2))*30)
              deg_y = int(((h/2 - center[1])/(w/2))*30)
              # print(deg_x,deg_y)
              # write_read(str(deg_x)+" "+str(deg_y))
            
              # Print the center point
              # print("Center Point: ({}, {})".format(center_x, center_y))


              # print(f"RBBox: {bb_box}")
      i+=1  
        
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# write_read("0 0")
# write_read("0 0")
# write_read("0 0")
# write_read("0 0")

# arduino.close()
cap.release()
cv2.destroyAllWindows()

