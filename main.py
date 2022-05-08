import cv2
import numpy as np
import sys
import os

sys.setrecursionlimit(1000000)


def read_image(filepath):
    return cv2.imread(filepath)


def show_image(window_name, image):
    cv2.imshow(window_name, image)


def image_conversion(image, formate):
    return cv2.cvtColor(image, formate)


def creating_empty_image(dim, dtype):
    return np.zeros(dim, dtype)


def image_contrast_brightness(image):
    new_image = creating_empty_image(image.shape, image.dtype)
    alpha = 4
    beta = -15
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y, x, 0] = np.clip(alpha * image[y, x, 0] + beta, 0, 255)
    return new_image


def image_scaling(image):
    scale_percent = min(35000 / image.shape[0], 200000 / image.shape[1])
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def image_sharpness(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


def print_stone(image, i, j, new_image):
    max_j = j
    if image[i][j] > 0:
        new_image[i][j] = image[i][j]
        image[i][j] = 0
        if 0 < i:
            max_j = max(max_j, print_stone(image, i - 1, j, new_image))
        if 0 < j:
            max_j = max(max_j, print_stone(image, i, j - 1, new_image))
        if i < image.shape[0] - 1:
            max_j = max(max_j, print_stone(image, i + 1, j, new_image))
        if j < image.shape[1] - 1:
            max_j = max(max_j, print_stone(image, i, j + 1, new_image))
    return max_j


def top_layer_stones(image, findTopStones=False):
    dim = image.shape
    new_image = creating_empty_image(dim, image.dtype)
    i = (len(image) // 2) if findTopStones else 0
    j = 0
    if i > 0:
        while j < dim[1]:
            if image[i][j] == 255:
                j = print_stone(image, i, j, new_image)
            j += 1
    else:
        while j < dim[1]:
            i = 0
            while i < dim[0]:
                if image[i][j] == 255:
                    j = print_stone(image, i, j, new_image)
                    break
                i += 1
            j += 1
    return new_image


def write_image(filename, image):
    if filename[-4:] != ".png":
        filename += ".png"
    if cv2.imwrite(os.getcwd() + "/" + filename, image):
        print(filename, "written successful")
    else:
        print(filename, " written failure")


def end_program():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = read_image("img1.png")
    img = image_scaling(img)
    show_image("image", img)
    img = image_conversion(img, cv2.COLOR_BGR2RGB)
    img = image_contrast_brightness(img)
    img = image_sharpness(img)
    img = image_conversion(img, cv2.COLOR_BGR2GRAY)
    dim = img.shape
    image_bw = creating_empty_image(dim, img.dtype)
    # setting threshold for the black and white image
    for i in range(dim[0]):
        for j in range(dim[1]):
            image_bw[i][j] = 255 if img[i][j] > 25 else 0

    all_solid_stones = top_layer_stones(image_bw)
    top_layer = top_layer_stones(all_solid_stones, True)

    show_image("result", top_layer)
    write_image("result", top_layer)
    end_program()


if __name__ == "__main__":
    main()
