#! /usr/bin/python

# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import smtplib
import RPi.GPIO as GPIO  # For Raspberry Pi GPIO control
import os

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# Use this xml file
cascade = "haarcascade_frontalface_default.xml"

# GPIO Pin for controlling the LED (you may need to change this)
LED_PIN = 21

# Set up the GPIO mode and LED pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# SMTP Email configuration
SMTP_SERVER = 'smtp.gmail.com'  # Change this to your SMTP server
SMTP_PORT = 587  # Change this to your SMTP server's port
SMTP_USERNAME = 'projectsagte@gmail.com'
SMTP_PASSWORD = 'noai meso okbo fqlb'
SENDER_EMAIL = 'projectsagte@gmail.com'
RECIPIENT_EMAIL = 'sreyesh.yenigalla23@gmail.com'

# Function for sending an email
def send_email(name, image):
    subject = "Unknown Visitor Detected"
    body = f"An unknown person is at your door."

    try:
        # Create an SMTP connection and send the email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)

        msg = f"Subject: {subject}\n\n{body}"
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg)
        server.quit()
        print(f"Email sent: {subject}")

        # Attach the image to the email
        send_image_email(image, name)

    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Function for sending an email with an attached image
def send_image_email(image, name):
    try:
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage
        import smtplib

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Unknown Visitor Detected"

        # Attach the image to the email
        img_data = open(image, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(image))
        msg.attach(image)

        body = f"An unknown person ({name}) is at your door."
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print(f"Email sent with image: Unknown Visitor Detected")

    except Exception as e:
        print(f"Error sending email with image: {str(e)}")

# Load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# Loop over frames from the video file stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select the first entry in the dictionary)
            name = max(counts, key=counts.get)

        # Update the list of names
        names.append(name)

        # If someone in your dataset is identified as "Unknown," send an email
        if name == "Unknown":
            # Take a picture to send in the email
            img_name = "image.jpg"
            cv2.imwrite(img_name, frame)
            print('Taking a picture.')

            # Now send an email to let you know an unknown person is at the door
            send_email(name, img_name)
            
        # If someone in your dataset is identified, turn on the LED
        if name != "Unknown":
            GPIO.output(LED_PIN, GPIO.LOW)
        else:
            GPIO.output(LED_PIN, GPIO.HIGH)

    # Loop over the recognized faces and draw bounding boxes
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

    # Display the image to the screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
GPIO.cleanup()  # Cleanup GPIO resources
