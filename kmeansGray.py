import cv2 as cv
import numpy as np
import math
import time
from random import randint


def euclideanDistance(pointA, pointB):
    channelsDifferencePower = math.pow(int(pointA) - int(pointB), 2)
    distance = math.sqrt(channelsDifferencePower)
    return distance


def centroidsRandomGeneration(img, kCentroids, clustersCentroids, clustersPoints, rows, cols):
    for i in range(kCentroids):
        centroidY = randint(0, rows - 1)
        centroidX = randint(0, cols - 1)
        centroidPoint = img[centroidY][centroidX]
        clustersCentroids.append(centroidPoint)
        clustersPoints.append([])
        clustersPoints[i].append(centroidPoint)


def centroidsPointsAssignment(img, clustersCentroids, centroidsPoints):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            minDistance = math.inf
            chosen = -1
            point = (row, col)
            for i in range(len(clustersCentroids)):
                centroid = clustersCentroids[i]
                distance = euclideanDistance(img[point[0]][point[1]], centroid)
                if distance < minDistance:
                    minDistance = distance
                    chosen = i
            centroidsPoints[chosen].append(point)


def adjustCentroidsPositions(img, clustersCentroids, clustersPoints):
    newDistance = 0.0
    nClusters = len(clustersCentroids)
    for i in range(nClusters):
        centroid = clustersCentroids[i]
        channelsMean = 0
        nPoints = len(clustersPoints[i])

        for point in clustersPoints[i]:
            channelsMean = channelsMean + img[point[0]][point[1]]
        oldCentroidValues = np.copy(centroid)
        if nPoints > 0:
            centroid = channelsMean / nPoints
        clustersCentroids[i] = centroid
        newDistance += euclideanDistance(oldCentroidValues, centroid)
    newDistance /= nClusters
    return newDistance


def applyFinalClusterToImage(img, clustersCentroids, clustersPoints):
    dst = np.copy(img)
    for i in range(len(clustersCentroids)):
        centroid = clustersCentroids[i]
        for point in clustersPoints[i]:
            dst[point[0]][point[1]] = centroid
    return dst


def kmeansgray(src, kCentroids):
    print("Start: %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    img = np.copy(src)
    print('Image Dimensions :', img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    clusterCentroids = []
    clustersPoints = []
    diffDistance = float('inf')
    shiftThreshold = 0.1
    oldDistance = float('inf')
    centroidsRandomGeneration(img, kCentroids, clusterCentroids, clustersPoints, rows, cols)
    while diffDistance > shiftThreshold:
        for i in range(len(clusterCentroids)):
            clustersPoints[i].clear()
        centroidsPointsAssignment(img, clusterCentroids, clustersPoints)
        newDistance = adjustCentroidsPositions(img, clusterCentroids, clustersPoints)
        diffDistance = abs(oldDistance - newDistance)
        oldDistance = newDistance
        print("diff: ", diffDistance)
    outputImage = applyFinalClusterToImage(img, clusterCentroids, clustersPoints)
    exec_time = time.time() - start_time
    print("\tSecondi: %s" % exec_time)
    print("Finish: %s" % (time.asctime(time.localtime(time.time()))))
    return [np.copy(outputImage), str(exec_time)]