import cv2

'''
# Image path
imageFile = "sources/car-image.jpg"
'''

#Video Path
videoFile = "sources/Cars_Pedestrians.mp4"

# Pre-Trained car-image Classifier
car_classifier_file = "car_detector.xml"

#Pre-Trained Fullbody classifier
pedestrian_classifier_file = "fullbody.xml"

'''
# Create OpenCV image
img = cv2.imread(imageFile)
'''

#Cv2 Video Capture
video = cv2.VideoCapture(videoFile)

#Create car-classififer
car_tracker = cv2.CascadeClassifier(car_classifier_file)

#Create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

#Run until car stops.
while True:
    #Read Current Frame
    (read_successful,frame) = video.read()

    if read_successful:
        #convert to grayscale
        grayscale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    '''
    #Convert to grayscale (needed for haar cascade)
    black_and_white_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    '''

    #Detect Cars
    cars = car_tracker.detectMultiScale(grayscale_frame)

    #Detect Pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    #Draw Rectangles around the car
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Draw Rectangles around the Pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)


    # display the frame with cars and pedestrian detected
    cv2.imshow("Car and Pedestrian detector",frame)

    # Don't autoclose and wait (here for a key press)
    key = cv2.waitKey(1)

    #Stop if Q is pressed
    if key == 81 or key == 113:
        break

#Release the VideoCapture Object
video.release()

print("Code Completed")

# ! The Commented Section is for still image detection