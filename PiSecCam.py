#!/usr/bin/env python3

from picamera.array import PiRGBArray
from io import BytesIO
from threading import Thread, Lock
import picamera
import argparse
import datetime
import json
import time
import cv2
import sys
import shutil

class Detector():
    def __init__(self, camera, streamer):
        self.camera = camera
        self.streamer = streamer
        self.arrayCapture = picamera.array.PiRGBArray(camera,
                size=tuple(conf["detection_resolution"]))
        self.lock = Lock()

    def shutdown(self):
        self.arrayCapture.seek(0)
        self.arrayCapture.truncate()

    def analyze(self):
        print("Analyze")
        avg = None
        lastMotionTime = datetime.datetime.now()
        stopRecTime = datetime.datetime.now()
        previouslyRecording = False

        stream = self.camera.capture_continuous(self.arrayCapture, format="bgr",
                use_video_port=True, resize=tuple(conf["detection_resolution"]))

        for f in stream:
            #grab the raw NumPy array representing the image and initialize the timestamp
            frame = f.array
            timestamp = datetime.datetime.now()
            motionDetected = False

            #resize the frame, convert it to grayscale, and blur the noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            #if the average frame is None, initialze it
            if avg is None:
                avg = gray.copy().astype("float")
                self.arrayCapture.seek(0)
                self.arrayCapture.truncate(0)
                continue

            #accumulate the weighted average between the current frame and previous,
            #then compute the difference between the frame and avg
            cv2.accumulateWeighted(gray, avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

            #threshold delta image, dilate the threshold image to fill in holes,
            #then find contours on threshold image
            threshold = cv2.threshold(frameDelta, conf["delta_threshold"], 255,
                    cv2.THRESH_BINARY)[1]
            threshold = cv2.dilate(threshold, None, iterations=2)
            (_, contours, _) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

            #loop over contours
            for c in contours:
                #if the contour is too small, ignore it
                if cv2.contourArea(c) < conf["min_area"]:
                    continue

                #computer the bounding box for the contour and draw it
                if conf["show_bounding_box"]:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                motionDetected = True

            #print(motionCounter)

            if motionDetected:
                ts = timestamp.strftime("%A %d %B %Y %I:%M%S%p")
                #cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                #increment motion frame counter
                motionCounter += 1

                #check to see if the number of frames is high enough to save vid
                if motionCounter >= conf["min_motion_frames"]:
                    lastMotionTime = timestamp
                    if not previouslyRecording:
                        if (timestamp - stopRecTime).seconds > conf["min_record_delay"]:
                            self.streamer.motionDetected()
                            previouslyRecording = True
            else:
                motionCounter = 0
                if previouslyRecording:
                    if (timestamp - lastMotionTime).seconds > conf["video_padding_seconds"]:
                        self.streamer.motionEnded(timestamp.strftime("%b-%d-%Y %I-%M-%S%p"))
                        stopRecTime = timestamp
                        previouslyRecording = False

            #check to see if user wants frames to be displayed
            if conf["show_video"]:
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                #if Q key is pressed, close program
                if key == ord("q"):
                    return

            self.arrayCapture.truncate(0)

class Streamer():
    def __init__(self, camera):
        self.camera = camera
        self.padding = conf["video_padding_seconds"]
        self.memoryStream = picamera.PiCameraCircularIO(self.camera,
            seconds = self.padding)
        self.motionStream = BytesIO()
        self.camera.start_recording(self.memoryStream, format="h264")

    def motionDetected(self):
        self.memoryStream.copy_to(self.motionStream, seconds = self.padding)
        self.camera.split_recording(self.motionStream)
        self.memoryStream.clear()

    def motionEnded(self, timestamp):
        self.camera.split_recording(self.memoryStream)
        self.motionStream.seek(0)
        with open(conf["video_save_location"]+timestamp+".mkv", "wb") as file:
            shutil.copyfileobj(self.motionStream, file)
        self.motionStream.seek(0)
        self.motionStream.truncate()

    def shutdown(self):
        self.camera.stop_recording()
        self.memoryStream.clear()
        self.motionStream.seek(0)
        self.motionStream.truncate()

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=False,
    help="path to the JSON file of configuration default settings")
args = vars(ap.parse_args())

conf_file = args["conf"]
if not conf_file:
    conf_file = "conf.json"
try:
    conf = json.load(open(conf_file))
except:
    print("Specify a configuration file with --conf or make sure it is proper JSON")
    sys.exit(1)

with picamera.PiCamera() as camera:
    camera.resolution = tuple(conf["video_resolution"])
    #camera.framerate = conf["fps"]

    streamer = Streamer(camera)
    detector = Detector(camera, streamer)

    #allow camera to warmup, then initialize the average frame, last save time,
    #and frame motion counter
    print("Warming up")
    time.sleep(conf["camera_warmup_time"])

    detector.analyze()

    detector.shutdown()
    streamer.shutdown()
