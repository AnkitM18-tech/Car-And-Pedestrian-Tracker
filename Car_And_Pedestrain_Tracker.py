import cv2

# Image path
imageFile = "sources/car-image.jpg"

# Pre-Trained car-image Classifier
classifier_file = "car_detector.xml"

# Create OpenCV image
img = cv2.imread(imageFile)

#Create car-classififer
car_tracker = cv2.CascadeClassifier(classifier_file)

# display the image with faces detected
cv2.imshow("Car and Pedestrian detector",img)

# Don't autoclose and wait (here for a key press)
cv2.waitKey()

print("Code Completed")