import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
import mxnet as mx

class Video:
    """
    Reader for video files
    """

    def __init__(self, videoPathArray):
        # self.lstFilenameOut = lstFilenameOut
        self.videoPathArray = videoPathArray #[['pathToVideo', 0], ['pathToVideo', 1]]
        self.videoStart = 500
        self.bundleSize = 8
        self.bundleNumber = 0
        self.imageResizerRatio = 4
        self.videoData = []
        self.mxNetArraysData = []
        self.mxNetArraysLabels = []
        self.frameCount = 1200
        self.packSize = 0
    
    def start_here(self):
        i = 0
        while i < len(self.videoPathArray): 
            self.mxNetArraysData.append(self.process_video(self.videoPathArray[i]))
            self.mxNetArraysLabels.append(mx.nd.full((self.packSize, 1), self.videoPathArray[i][1]))
            i += 1

        data, label = self.concat_arrays(self.mxNetArraysData, self.mxNetArraysLabels)
        train_data, val_data = self.define_iterator(data, label)

        print('------ Train_data: ', train_data)
        print('------ Val_data: ', val_data)
        return train_data, val_data

    
    def concat_arrays(self, dataArray, labelArray):
        print('------ Shape: ', dataArray[0].shape)
        print('------ Shape: ', dataArray[1].shape)

        combinedData = mx.nd.concat(dataArray[0], dataArray[1], dim=0)
        print('------ CombinedData shape: ', combinedData.shape)

        print('------ Shape: ', labelArray[0].shape)
        print('------ Shape: ', labelArray[1].shape)

        combinedLabel = mx.nd.concat(labelArray[0], labelArray[1], dim=0)
        print('------ CombinedLabel shape: ', combinedLabel.shape)

        return combinedData, combinedLabel

    def process_video(self, currentVideoCap):
        currentCap = cv2.VideoCapture(currentVideoCap[0])

        #needed to offset the video by self.videoStart = 500 frames
        captures = 0
        while captures < self.videoStart:
            ret, currentFrame = currentCap.read()
            captures += 1

        frameCount = self.frameCount #for testing purposes
        # frameCount_real = int(currentCap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(currentCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(currentCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        newWidth = int(frameWidth // self.imageResizerRatio)
        newHeight = int(frameHeight // self.imageResizerRatio)

        self.packSize = frameCount - self.videoStart - self.bundleSize + 1
        
        newMxNetArray = mx.nd.empty((self.packSize, 24, newWidth, newHeight))

        fc = 0
        newFc = 0
        ret = True
        # arrays used for iteration
        currentMxArray = mx.nd.empty((self.bundleSize, newHeight, newWidth, 3))
        currentNonMxArray = []

        while fc < self.packSize + self.bundleSize and ret:
            ret, currentFrame = currentCap.read()
            currentFrame = self.crop_image(currentFrame, frameHeight, frameWidth, newHeight, newWidth)
            
            if len(currentNonMxArray) < self.bundleSize:
                currentNonMxArray.append(currentFrame)
            else:
                # Turn the temp array into mxnet Array with values between 0 and 1
                currentMxArray = mx.nd.array(currentNonMxArray).astype(np.float32)/255 
                if(newFc >= self.packSize):
                    break
                newMxNetArray[newFc] = currentMxArray[0:self.bundleSize, :, :, :].transpose((0, 3, 2, 1)).reshape((-1, 320, 180)) # reshaping
                currentNonMxArray.pop(0) # Remove the first item of the list
                currentNonMxArray.append(currentFrame)
                newFc += 1
            fc += 1

        return newMxNetArray
    
    def crop_image(self, img, frameHeight, frameWidth, cropSizeY, cropSizeX):
        """
        crop_image(self, img, frameHeight, frameWidth, cropsizeY newcropSizeX)

        Crops images to the default blah lbah
        """
        startx = frameWidth//2 - cropSizeX//2
        starty = frameHeight//2 + cropSizeY//2
        return img[starty:starty+cropSizeY, startx:startx+cropSizeX]

    def define_iterator(self, data, label):
        batch_size = 32
        permutation = np.random.permutation(len(label))
        p_data = data[permutation]
        p_label = label[permutation]
        ntrain = int(data.shape[0] * 0.8)
        
        train_iter = mx.io.NDArrayIter(p_data[:ntrain, :], p_label[:ntrain], batch_size, shuffle=True)
        val_iter = mx.io.NDArrayIter(p_data[ntrain:, :], p_label[ntrain:], batch_size)
        return train_iter, val_iter

    # if __name__ == '__main__':
    #     main(self)