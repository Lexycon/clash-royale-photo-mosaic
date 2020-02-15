import cv2
import numpy as np
import os
import random
import math
import multiprocessing
import sys
import time


class ImageProcessor:
    def __init__(self, tileSizes):
        # frames are 1080x2340 (ONEPLUS 7T)
        # define ROI, height between pixel 470 and 1950
        # so cut away 470 top and 390 (2340-1950) bottom
        self.startX = 0
        self.startY = 470
        self.endX = 1080
        self.endY = 1950

        self.dirFrames = 'frames'
        self.dirFramesModified = 'frames_modified'
        self.dirTemplates = 'templates'
        self.dirTiles = 'tiles'

        # add tile sizes which will be generated
        self.tileSizes = tileSizes

        # tiles will be added to this, to check for similarity later
        self.imageTiles = {}

        # new tiles will be added to a buffer
        # waiting until this buffer equals cpu core count to calc sim
        self.imageTilesBuffer = {}

        # generate empty arrays for dict / each tile size
        for tileSize in self.tileSizes:
            self.imageTiles[str(tileSize)] = []
            self.imageTilesBuffer[str(tileSize)] = []

        # loading matching templates
        self.templates = self.__loadingTemplates()

        # get cpu cores count
        self.procCount = multiprocessing.cpu_count()

        # define dict for multiprocessing to collect return values of processes
        manager = multiprocessing.Manager()
        self.procDict = manager.dict()

        # mean square error threshold of image comparsion
        self.mseThreshold = 5000

    def __mse(self, imageA, imageB):
        # mean square error
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def __loadingTemplates(self):
        templates = []
        for filename in sorted(os.listdir(self.dirTemplates)):
            if filename.endswith('.jpg'):
                template = cv2.imread(os.path.join(self.dirTemplates, filename))
                templates.append(template)

        return templates

    # simultaneously executed
    def __workCheckSimilarity(self, procNum, imgTile, tileSize):
        # check similarity with all existing tiles
        similar = False

        for tile in self.imageTiles[str(tileSize)]:
            ret = self.__mse(imgTile[1], tile)
            if (ret < self.mseThreshold):
                similar = True
                break

        # tile is not similar with any existing tile
        if (not similar):
            self.procDict[procNum] = imgTile

    def __clearTileBuffer(self, tileSize):
        procs = []

        # each tile in buffer will loop all tiles saved until now on own cpu core
        for tile in self.imageTilesBuffer[str(tileSize)]:
            proc = multiprocessing.Process(
                target=self.__workCheckSimilarity, args=(len(procs), tile, tileSize))

            # add process to process list
            procs.append(proc)

        # lets do some work, push the ryzen
        for proc in procs:
            proc.start()

        # Wait for all processes to complete
        for proc in procs:
            proc.join()

        # get tiles which are not duplicated, ready to add them
        procBuffer = self.procDict.values()

        # clean up processes
        procs = []
        self.procDict.clear()

        # create tiles and add them to list
        for imgTile in procBuffer:
            # write tile
            cv2.imwrite(imgTile[0], imgTile[1])
            # add to tiles to be in list for upcoming tiles
            self.imageTiles[str(tileSize)].append(imgTile[1])

        # clear up buffer
        self.imageTilesBuffer[str(tileSize)] = []

        self.procDict.clear()

    def __addTileToBuffer(self, outFile, imgTile, tileSize):
        # check similarity of tile with tiles already in buffer to be sure they are unique
        # because they will run parallel later
        similar = False

        for tile in self.imageTilesBuffer[str(tileSize)]:
            ret = self.__mse(tile[1], imgTile)
            if (ret < self.mseThreshold):
                similar = True
                break

        # tile not similar with any tile in buffer, potential candidate, else throw away
        if (not similar):
            self.imageTilesBuffer[str(tileSize)].append([outFile, imgTile])

            # if we have 1 tile for each core, fire up to clean buffer
            if (self.procCount == len(self.imageTilesBuffer[str(tileSize)])):
                self.__clearTileBuffer(tileSize)


    # simultaneously executed
    def __workTemplateMatching(self, procNum, template, imgModified):
        # will calc points where template is found (where a unit is)
        pointsMatchingTemplate = []

        # size of template image, used for drawing rectangle later in modified frame
        templateShape = list(template.shape[:-1])

        # matching with template image
        res = cv2.matchTemplate(imgModified, template, cv2.TM_CCOEFF_NORMED)
        # threshold .85
        loc = np.where(res >= .85)

        # lots of matched points, reducing this so no point
        # will be next to each other with a distance of 100px
        # else lots of matches, see skeleton army...
        # which will result in much skeleton tiles (which will it anyway still :))
        for point in zip(*loc[::-1]):
            similar = False
            if pointsMatchingTemplate:
                for pointMatch in pointsMatchingTemplate:

                    distance = math.sqrt(math.pow((point[0]-pointMatch[0]), 2) +
                                         math.pow((point[1]-pointMatch[1]), 2))

                    if (distance < 100.0):
                        similar = True
                        break

            if not similar:
                pointsMatchingTemplate.append(point)

        # write result to processes share dict
        self.procDict[procNum] = [pointsMatchingTemplate, list(templateShape)]

    def createTiles(self):
        print('Creating tiles...')
        start = time.time()
        random.seed(1337)

        # define output image to see the cores working
        cv2.namedWindow('tiles', cv2.WINDOW_NORMAL)

        frames = sorted(os.listdir(self.dirFrames))

        for num, filename in enumerate(frames):
            if filename.endswith('.jpg'):
                # print
                sys.stdout.write('{}/{}\r'.format(num, len(frames)))
                sys.stdout.flush()

                # read frame
                imgOriginal = cv2.imread(
                    os.path.join(self.dirFrames, filename))

                # copy original one for cut tiles later, modified will be used for rectangle drawing, ROI etc
                imgModified = imgOriginal.copy()

                # set the ROI, cut off user card board etc
                imgModified = imgModified[self.startY:self.endY,
                                          self.startX:self.endX]

                # remove health bar pixels of the 4 princess towers, else they get detected :(
                imgModified[200:250, 150:300] = 0
                imgModified[200:250, 775:925] = 0
                imgModified[1200:1250, 150:300] = 0
                imgModified[1200:1250, 775:925] = 0

                # define empty processes list
                procs = []

                # for each template run cpu
                for template in self.templates:
                    proc = multiprocessing.Process(target=self.__workTemplateMatching, args=(
                        len(procs), template, imgModified))

                    procs.append(proc)

                # Start all processes
                for proc in procs:
                    proc.start()

                # Wait for all processes to complete
                for proc in procs:
                    proc.join()

                # get data from processes
                procBuffer = self.procDict.values()

                # clean up processes
                procs = []
                self.procDict.clear()

                # sum up all points from each cpu calculated
                # adding templatesize here is just to know which point was found by which template
                # to draw the rectangle later, bit overhead, could be solved better, but...
                pointsMatched = []
                for points, templateSize in procBuffer:
                    for point in points:
                        pointsMatched.append([point, templateSize])

                # for each frame start at zero
                tileCounter = 0

                # if template matching detected some points
                if pointsMatched:

                    # loop the points
                    for (pointX, pointY), (templateSizeY, templateSizeX) in pointsMatched:
                        # draw rectangle about the matching
                        cv2.rectangle(imgModified, (pointX, pointY), (pointX + templateSizeX, pointY + templateSizeY), (0, 255, 0), 2)

                        # if multiple tilesizes (48, 64, ...) should be extracted
                        for tileSize in self.tileSizes:

                            # generate a random for the width, the card level will be centered if unit is not damaged
                            # if unit is damaged it will move to top left and a healthbar will be shown
                            # i choosed the lucky way, a bit random in the middle.
                            rndm = random.randint(
                                int(tileSize / 1.7), int(tileSize / 1.2))
                            pointTileCutX = int(
                                pointX + (templateSizeX / 2) - (rndm))

                            # if we moved to far left to cut out, because the matching was close to left border
                            if (pointTileCutX < 0):
                                pointTileCutX = 0
                            # if we are too far right...
                            if (pointTileCutX > (self.endX - (tileSize*2))):
                                pointTileCutX = int(self.endX - (tileSize*2))

                            # rndm = random.randint(int(tileSize / 6), int(tileSize / 4))
                            # pointTileCutY = int(point[1] + (h / 2) - rndm)
                            # start cut out height below the matched level of the unit
                            # idea is to remove the amount of "card levels" shown in tiles
                            pointTileCutY = int(
                                (pointY + self.startY) + templateSizeY)

                            # cut should be smaller than image height...
                            if (pointTileCutY > (self.endY - (tileSize*2))):
                                pointTileCutY = int(self.endY - (tileSize*2))

                            # cut out tile of original image with calculated x and y (pointTileCutX, pointTileCutY)
                            imgTile = imgOriginal[pointTileCutY:pointTileCutY+(
                                tileSize*2), pointTileCutX:pointTileCutX+(tileSize*2)]

                            # drawing cut out rectangle on modified image
                            cv2.rectangle(imgModified, (pointTileCutX, pointTileCutY-self.startY), (pointTileCutX+(
                                tileSize*2), pointTileCutY+(tileSize*2)-self.startY), (255, 255, 255), 2)

                            # resize the cutted out tile to half size. (e.g. for 48x48 tiles we cut out 96x96)
                            # i think this improves quality of the smaller tiles?!
                            imgTile = cv2.resize(
                                imgTile, (0, 0), fx=0.5, fy=0.5)

                            # some tiles have black borders left/right, because CR scales bad on my OP7 with 21:9
                            # so i drop tiles with an huge black average color on left/right
                            imgTileGrey = cv2.cvtColor(
                                imgTile, cv2.COLOR_BGR2GRAY)
                            imgAvgColorLeft = imgTileGrey[0:tileSize, 0:2].mean(axis=0).mean(axis=0)
                            imgAvgColorRight = imgTileGrey[0:tileSize, (tileSize-2):].mean(axis=0).mean(axis=0)

                            if (imgAvgColorLeft > 5 and imgAvgColorRight > 5):
                                outFile = os.path.join(self.dirTiles + '{}'.format(tileSize),
                                    filename[:-4] + '_{:d}.jpg'.format(tileCounter))

                                # tile potential candidate, add to buffer, similarity check needs to be done later
                                self.__addTileToBuffer(outFile, imgTile, tileSize)

                        tileCounter += 1

                # write modified image (contains rectangle template match + rectangle tile cut out)
                cv2.imwrite(os.path.join(self.dirFramesModified, filename), imgModified)

                # show
                cv2.imshow('tiles', imgModified)
                cv2.waitKey(1)

        # clean last tiles in buffer
        for tileSize in self.tileSizes:
            self.__clearTileBuffer(tileSize)

        end = time.time()
        print("Tiles created in {} seconds".format(int(end-start)))
