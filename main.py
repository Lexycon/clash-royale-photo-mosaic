from VideoProcessor import VideoProcessor
from ImageProcessor import ImageProcessor
from MosaicProcessor import MosaicProcessor

if __name__ == '__main__':
    ### video to frames
    ### (skipFramesBeginEnd, stepFrames)

    videoProcessor = VideoProcessor(300, 45)
    videoProcessor.createFrames()

    # ### frames to tiles
    # ### ([tileSizes])

    imageProcessor = ImageProcessor([48, 64])
    imageProcessor.createTiles()

    ### tiles to mosaic
    ### (inFile, outFile, tilesDirectory, outFileTilesInXAxis)

    mosaicProcessor = MosaicProcessor('cr.jpg', 'out.jpg', 'tiles48-lexycon', 84)
    mosaicProcessor.createMosaic()
