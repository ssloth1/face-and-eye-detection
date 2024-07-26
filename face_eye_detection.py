# James Bebarski
# Face & Eye Detection
# July 3, 2024

import cv2
import numpy as np

class FaceDetector:
    
    # reads in the face and eye cascade classifiers, sunglasses image, and initializes the face_positions list
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.sunglasses_img = cv2.imread('dealwithit.png', cv2.IMREAD_UNCHANGED)
        self.face_positions = []

    # detects faces in the frame, returns the face positions
    def detect_faces(self, gray): 
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # using the face measurements, get a better estimate of the eye's positions relative to the face
    def detect_eyes(self, gray, face_measurements):
        (x, y, width, height) = face_measurements
        return self.eye_cascade.detectMultiScale(gray[y : y + height, x : x + width], scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    # draws the detected faces and eyes on the given frame. As an option, sunglasses can be overlayed on top of the detected faces
    def draw_detections(self, frame, gray, faces, sunglasses_enabled):

        # while the number of detected faces is greater than the number of stored face positions, append an empty list
        # the list is to keep track of the face positions for averaging, I found that averaging the face positions over a few frames helped with the jittering        
        while len(self.face_positions) < len(faces):
            self.face_positions.append([]) 
        
        # draw rectangles around the faces in green and overlay sunglasses
        for i, (x, y, width, height) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # append the face positions to the list
            self.face_positions[i].append((x, y, width, height))

            # Keep only the last 5 positions for averaging
            self.face_positions[i] = self.face_positions[i][-5:]

            # overlay sunglasses if enabled, otherwise draw the red rectangles around the eyes
            if sunglasses_enabled:
                avg_face_position = np.mean(self.face_positions[i], axis=0).astype(int)
                self.overlay_sunglasses(frame, *avg_face_position)
            else:
                eyes = self.detect_eyes(gray, (x, y, width, height))
                for (eye_x, eye_y, eye_width, eye_height) in eyes:
                    cv2.rectangle(frame, (x + eye_x, y + eye_y), (x + eye_x + eye_width, y + eye_y + eye_height), (0, 0, 255), 2)
        
        return frame
    
    # overlays the sunglasses on the detected face
    def overlay_sunglasses(self, frame, x, y, width, height):

        # resize the sunglasses image to fit the detected face width
        sunglasses_width = width
        aspect_ratio = self.sunglasses_img.shape[0] / self.sunglasses_img.shape[1]
        sunglasses_height = int(sunglasses_width * aspect_ratio)
        resized_sunglasses_img = cv2.resize(self.sunglasses_img, (sunglasses_width, sunglasses_height))
        
        # I had to create a slight offset to make the sunglasses fit the face better
        offset = int(height * 0.1)  
        y1, y2 = y + int(height/4) + offset, y + int(height/4) + sunglasses_height + offset
        x1, x2 = x, x + sunglasses_width
        
        # check if the region of interest is within the frame
        if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return
        
        region = frame[y1:y2, x1:x2]
        
        # split the resized sunglasses image into its color and alpha channels
        sunglasses_color = resized_sunglasses_img[:, :, :3]
        sunglasses_alpha = resized_sunglasses_img[:, :, 3] / 255.0
        
        # blend the sunglasses image with the region that was detected as a face
        for i in range(0, 3):
            region[:, :, i] = (1.0 - sunglasses_alpha) * region[:, :, i] + sunglasses_alpha * sunglasses_color[:, :, i]

def main():

    # initialize the face detector and the webcam
    face_detector = FaceDetector()
    cap = cv2.VideoCapture(0)    
    print("Initializing Webcamera...")
    print("Press any key to exit.")
    if not cap.isOpened():
        print("Error: Could not access your webcamera.")
        return


    cv2.namedWindow('Face and Eye Detection')
    
    # creates a trackbar to enable/disable the sunglasses overlay
    sunglasses_enabled = 0
    def on_trackbar(val):
        nonlocal sunglasses_enabled
        sunglasses_enabled = val
    
    cv2.createTrackbar('Deal With It', 'Face and Eye Detection', 0, 1, on_trackbar)
    
    # read until the user presses any key
    while True:
   
        val, frame = cap.read()
        
        if not val:
            print("Error: Could not read frame.")
            break
        
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Face and Eye Detection', face_detector.draw_detections(frame, gray, face_detector.detect_faces(gray), sunglasses_enabled))
        
        # break the loop if the user presses any key
        if cv2.waitKey(1) != -1:
            print("Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()