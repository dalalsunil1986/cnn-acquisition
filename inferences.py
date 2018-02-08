# Load the model 

# Load the video using opencv

# Start loop here
# package 8 frames into a dataIter
# send the frames to the model to get a prediction
# Do it again

import cv2
import numpy as np
import mxnet as mx

class PredictFromVideo:
    def __init__(self, pathToVideo):
        # self.pathToModel = pathToModel
        self.cap = cv2.VideoCapture(pathToVideo)
        self.bundleSize = 8
        self.imageResizerRatio = 4
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.newWidth = int(self.frameWidth // self.imageResizerRatio)
        self.newHeight = int(self.frameHeight // self.imageResizerRatio)

        # load the model
    
    def packageImages(self):
        # Every seconds: Fill out an array with with the capture
        # send the array to arrayMxArray
        ret = True
        idx = 0
        imageArray = []
        while ret and idx < self.frameCount - self.bundleSize + 1:
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)
            # & 0xFF is required for a 64-bit system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(imageArray) < self.bundleSize:
                imageArray.append(self.crop_image(frame))
            else:
                self.arrayMxArray(imageArray)
                # break
                imageArray = []
            # print('length :', len(imageArray))
            idx += 1

    def crop_image(self, img):
        startx = self.frameWidth//2 - self.newWidth//2
        starty = self.frameHeight//2 + self.newHeight//2
        return img[starty:starty+self.newHeight, startx:startx+self.newWidth]

    def arrayMxArray(self, array):
        # turn the 8 Packs array into an mxnet iterator
        mxArray = mx.nd.array(array).astype(np.float32)/255
        mxArray = mxArray[0:self.bundleSize, :, :, :].transpose((0, 3, 2, 1)).reshape((-1, 320, 180)) # reshaping
        print(mxArray)

    def getInference(self, mxArray):
        print('nothing')

def main():
    # start the capture
    predict = PredictFromVideo('mk_video/fast_lap_01.mp4')
    predict.packageImages()

if __name__ == '__main__':
    main()


