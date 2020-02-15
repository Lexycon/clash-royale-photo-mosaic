import cv2
import numpy as np
import os
import math
import multiprocessing
import sys, time

class VideoProcessor:
    def __init__(self, skipFramesBeginEnd=300, stepFrames=45):
        # skipframes begin/end will remove first/last 5 seconds of video
        # just to wait a bit until cr game is loaded etc and cutting the end
        # because I switched to notifications to stop recording
        self.skipFramesBeginEnd = skipFramesBeginEnd

        # every skip frame a new 'screenshot/frame' will be taken
        self.stepFrames = stepFrames

        self.dirRecordings = 'recordings'
        self.dirFrames = 'frames'

    def createFrames(self):
        print('Creating frames...')
        start = time.time()
        frameNameCounter = 0

        recordings = sorted(os.listdir(self.dirRecordings))
        for num, filename in enumerate(recordings):
            if filename.endswith('.mp4'):
                # print
                sys.stdout.write('{}/{}\r'.format(num, len(recordings)))
                sys.stdout.flush()

                cap = cv2.VideoCapture(os.path.join(
                    self.dirRecordings, filename))
                property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                length = int(cv2.VideoCapture.get(cap, property_id))
                frameCounter = 0

                ret, frame = cap.read()

                # default 300 => skip first 5 seconds 60 fps
                frameCounter += self.skipFramesBeginEnd
                cap.set(1, frameCounter)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(os.path.join(
                            self.dirFrames, 'frame{:d}.jpg'.format(frameNameCounter)), frame)
                        frameNameCounter += 1

                        # default 45 => skip .75 second (60fps) (real x4 speed replay 3 seconds)
                        frameCounter += self.stepFrames

                        if (frameCounter >= (length-self.skipFramesBeginEnd)):
                            cap.release()
                            break
                        cap.set(1, frameCounter)
                    else:
                        cap.release()
                        break

        end = time.time()
        print("Frames created in {} seconds".format(int(end-start)))
