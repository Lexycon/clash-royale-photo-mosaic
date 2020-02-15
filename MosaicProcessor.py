import cv2
import numpy as np
import os
import math
import multiprocessing
import sys, time


class MosaicProcessor:
    def __init__(self, inFile, outFile, dirTiles='tiles48-lexycon', outTileCountX=64):
        self.inFile = inFile
        self.outFile = outFile

        self.dirTiles = dirTiles

        # get first tile file to get the dimension of the tiles
        self.outTileSize = cv2.imread(os.path.join(dirTiles, os.listdir(dirTiles)[0])).shape[0]

        # calc images sizes, dimension of out image
        # split in image in different tiles, etc.
        self.__calcImageProperties(outTileCountX)

        # loading tiles
        self.imgTiles = self.__loadImageTiles()

        # get cpu cores count
        self.procCount = multiprocessing.cpu_count()

        # define dict for multiprocessing to collect return values of processes
        manager = multiprocessing.Manager()
        self.procDict = manager.dict()

    def __calcImageProperties(self, outTileCountX):
        self.outTileCountX = outTileCountX

        # read file and get x and y
        self.inImg = cv2.imread(self.inFile)
        self.inY, self.inX, _ = self.inImg.shape

        # calc width of outFile
        self.outX = self.outTileSize * self.outTileCountX

        # define tile size of inFile
        self.inTileSize = self.inX / self.outTileCountX
        self.inTileCountX = int(self.inX / self.inTileSize)
        self.inTileCountY = int(self.inY / self.inTileSize)
        self.inTileSize = int(self.inTileSize)

        # inFile will be splitted in same tile count than outFile, but this couldn't fit perfectly
        # so maybe we won't use the full inFile image size
        self.inUsingX = self.inTileCountX * self.inTileSize
        self.inUsingY = self.inTileCountY * self.inTileSize

        # calc height of outFile
        self.outY = self.outTileSize * self.inTileCountY

        # calc tile count x*y
        self.sumTileCount = self.inTileCountX * self.inTileCountY

        print('IN  Resolution: {}x{}'.format(self.inX, self.inY))
        print('IN  TileSize  : {}x{}'.format(self.inTileSize, self.inTileSize))
        print('IN  Using Res.: {}x{}\n'.format(self.inUsingX, self.inUsingY))
        print('OUT Resolution: {}x{}'.format(self.outX, self.outY))
        print('OUT TileSize  : {}x{}\n'.format(self.outTileSize, self.outTileSize))
        print('Tiles to calc : {} ({}x{})'.format(self.sumTileCount, self.inTileCountX, self.inTileCountY))

    def __loadImageTiles(self):
        # define tiles directory

        imgTiles = []
        for filename in os.listdir(self.dirTiles):
            if filename.endswith('.jpg'):
                # read tile
                imgTile = cv2.imread(os.path.join(self.dirTiles, filename))

                # resize tile to inFile tile size, to compare them later
                imgTileResize = cv2.resize(imgTile, (self.inTileSize, self.inTileSize))

                imgTiles.append([imgTile, imgTileResize])

        return imgTiles

    def __mse(self, imageA, imageB):
        # mean square error
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    # simultaneously executed
    def __work(self, procNum, tileNum, scanTile):
        bestCandidiates = []
        for num, tile in enumerate(self.imgTiles):
            # mse
            err = self.__mse(scanTile, tile[1])

            # find best candidiate tiles (as many as cpu core counts)
            # because other cores could find the same tile and we don't want duplication
            # every tile is better than none, so just fill up the first matches
            if (len(bestCandidiates) < self.procCount):
                bestCandidiates.append([err, num])

            # check for better candidiates, replace the "worst" on idx zero
            elif (err < bestCandidiates[0][0]):
                bestCandidiates[0] = [err, num]

            # worst candidate is always on idx zero, because of sorting
            bestCandidiates.sort(key=lambda x: x[0], reverse=True)

        # calc done, sort best candidate first, worst last
        bestCandidiates.sort(key=lambda x: x[0])

        # remap list, only tile idxs left, removed the mse error
        bestCandidiates = list(map(lambda l: l[1:2][0], bestCandidiates))

        # write result to processes share dict
        self.procDict[procNum] = [tileNum, bestCandidiates]

    def createMosaic(self):
        print('Creating mosaic...')
        start = time.time()

        # define empty processes list
        procs = []

        # generate empty image with height outY and width outX
        # careful, cv goes for y,x i always go for x,y...
        outImg = np.zeros((self.outY, self.outX, 3), np.uint8)

        # define output image to see the cores working
        cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)

        tileCounter = 0
        # looping the inFile Tiles
        for y in range(0, self.inUsingY, self.inTileSize):
            for x in range(0, self.inUsingX, self.inTileSize):

                # cut of current inFile tile (y,x)
                inImgTile = self.inImg[y:y +
                                       self.inTileSize, x:x+self.inTileSize]

                # generate process for this tile
                proc = multiprocessing.Process(target=self.__work, args=(
                    len(procs), tileCounter, inImgTile))

                # add process to process list
                procs.append(proc)

                tileCounter += 1

                # if we defined same process count than your cpu got cores
                # or we don't have any tiles left => fire up
                if (len(procs) == self.procCount or tileCounter == self.sumTileCount):

                    # lets do some work, push the ryzen
                    for proc in procs:
                        proc.start()

                    # Wait for all processes to complete
                    for proc in procs:
                        proc.join()

                    # no tile should be used twice, so mark them in a list
                    usedCandidiates = []

                    # get data from processes
                    procBuffer = self.procDict.values()
                    procBuffer.sort(key=lambda x: x[:][0])

                    # clean up processes
                    procs = []
                    self.procDict.clear()

                    # loop all data from processes with tile id and best matches found
                    for tileNumber, bestCandidates in procBuffer:

                        # each process calculated multiple candidates (as many as you got cpu cores)
                        # so in worst case each process calculated at least one unused tile
                        for bestCandidate in bestCandidates:

                            # if tile is not used by other processes
                            if bestCandidate not in usedCandidiates:

                                # calculate position of the outTile by tile id
                                temp_y = int(
                                    tileNumber / self.inTileCountX) * self.outTileSize
                                temp_x = (tileNumber %
                                          self.inTileCountX) * self.outTileSize

                                # add tile to used list
                                usedCandidiates.append(bestCandidate)

                                # add new tile to outImg
                                outImg[temp_y:temp_y + self.outTileSize, temp_x:temp_x +
                                       self.outTileSize] = self.imgTiles[bestCandidate][0]

                                break

                    # sort all used tiles highest first (easier to pop them later)
                    usedCandidiates.sort(reverse=True)

                    for usedCandidiate in usedCandidiates:
                        # delete tile from tile list, so it won't used again
                        self.imgTiles.pop(usedCandidiate)

                    # print
                    sys.stdout.write('{}/{}\r'.format(tileCounter, self.sumTileCount))
                    sys.stdout.flush()

                    # show
                    cv2.imshow('mosaic', outImg)
                    cv2.waitKey(1)


        end = time.time()
        print("Mosaic created in {} seconds".format(int(end-start)))
        cv2.imwrite(self.outFile, outImg)
        cv2.waitKey(0)


