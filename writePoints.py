import cv2
import torch
import matplotlib.pyplot as plt

def writePoints(source):
    '''
    Lets you write points onto an image. Returns the altered image along with the set of points you labeled.
    :param source: input image as tensor

    Example:
    image = read_image('./test_imgs/image.jpg')
    altered_img, point = writePoint(image)
    '''
    image = source.numpy()

    points = []

    def addPoint(action, x, y, flags, userdata):
        nonlocal points

        if action == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x,y), radius = 1, color=(0,255,0), thickness = 1)
            points.append((x,y))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', addPoint)
    while True: 
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27: 
            return image, points

if __name__ == '__main__':
    image = cv2.imread('./data/MNIST/overlapMNIST/train/61/13_61.png', cv2.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image)
    image, points = writePoints(image)
    print(points)