import cv2
import random
import numpy as np


# mean=[125.382, 119.98, 110.339], std=[41.446, 39.805, 40.64]
def normalize_img(img, mean=[123.67, 118.418, 108.34], std=[41.051, 39.251, 40.012]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1, img2, label = sample
        if random.random() < 0.5:
            #print("RandomHorizontalFlip")
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            label = cv2.flip(label, 1)

        return img1, img2, label

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1, img2, label = sample
        if random.random() < 0.5:
            #print("RandomVerticalFlip")
            img1 = cv2.flip(img1, 0)
            img2 = cv2.flip(img2, 0)
            label = cv2.flip(label, 0)

        return img1, img2, label

class RandomFixRotate(object):
    def __init__(self):
        self.degree=random.choice([90, 180, 270])
        #self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1, img2, label = sample
        if random.random() < 0.3:
            #print(self.degree)
            rows, cols = img1.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.degree, 1)
            img1 = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
            img2 = cv2.warpAffine(img2, rotation_matrix, (cols, rows))
            label = cv2.warpAffine(label, rotation_matrix, (cols, rows))
            # rotate_degree = random.choice(self.degree)
            # img1 = img1.transpose(rotate_degree)
            # img2 = img2.transpose(rotate_degree)
            # label = label.transpose(rotate_degree)

        return img1, img2, label