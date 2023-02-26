import cv2 as cv
import kmeans as km
import kmeansGray as kmGray
import numpy as np
from os import listdir
from os.path import isfile, join


def main():
    wantImgGray = False
    imagesPath = '../images/'
    imagesDirs = [f for f in listdir(imagesPath)]
    for imagesDir in imagesDirs:
        imagesDir = join(imagesPath, imagesDir)
        images = [f for f in listdir(imagesDir) if isfile(join(imagesDir, f))]
        for image_name in images:
            image_path = join(imagesDir, image_name)
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            color_string = 'color'
            if wantImgGray:
                color_string = 'gray'
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # img = np.stack((img,) * 3, axis=-1)

            img = cv.resize(img, (320, 240))
            cv.imshow('Original', img)
            cv.waitKey(0)
            repeat = 0
            kRegions = int(input('Enter number of regions: '))
            cv.destroyAllWindows()
            print('RegionsNumber: ', kRegions)
            while repeat < 5:
                repeat += 1
                if len(img.shape) > 2:
                    mspp_img = km.kmeans(img, kRegions)
                else:
                    mspp_img = kmGray.kmeansgray(img, kRegions)
                # repeat = int(input('WannaRepeat? 1Yes 0No: '))

                size_string = '320x240'
                filename = 'resultnt/TRYSLICING' + str(repeat) + color_string + size_string + image_name
                status = cv.imwrite(filename, mspp_img[0])
                print('Image written: %s - Status: %s' % (image_name, status))
                with open('resultnt/record.txt', 'a') as f:
                    exec_time = mspp_img[1]
                    result = str(repeat) + ' kRegions: ' + str(kRegions) + ' ' + image_name + ': ' + str(exec_time) + '\n'
                    f.write(result)
                    f.close()
    '''img = cv.imread('../images/dataset20/vegetables.JPG', cv.IMREAD_COLOR)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv.resize(img, (320, 240))
    # img = img.reshape((-1, 3))
    k = 8
    kmeaned = km.kmeans(img, k)
    cv.imshow('Output', kmeaned)
    cv.waitKey(0)'''


if __name__ == '__main__':
    main()
