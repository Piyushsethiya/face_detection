import cv2
import urllib.request

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = "people_image.jpg"

# Download the image from URL 
# You can Customized Path 
urllib.request.urlretrieve("https://img.freepik.com/free-photo/community-concept-with-group-people_23-2147993334.jpg", path)
# Read the input image
img = cv2.imread (path)

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load image.")
    exit(1)
    
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()