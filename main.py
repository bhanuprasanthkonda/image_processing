import cv2
import numpy as np
import sys
import os

sys.setrecursionlimit(1000000)


# reads the image from the filepath
def read_image(filepath):
    return cv2.imread(filepath)


# show the image
def show_image(window_name, image):
    cv2.imshow(window_name, image)


# returns a converted image
def image_conversion(image, formate):
    return cv2.cvtColor(image, formate)


# returns a new empty image
def creating_empty_image(dim, dtype):
    return np.zeros(dim, dtype)


# setting the contras and brightness of the image
def image_contrast_brightness(image):
    new_image = creating_empty_image(image.shape, image.dtype)
    alpha = 4
    beta = -15
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y, x, 0] = np.clip(alpha * image[y, x, 0] + beta, 0, 255)
    return new_image


# scaling the image to (350,) or (0,2000) dimensions
def image_scaling(image):
    scale_percent = min(35000 / image.shape[0], 200000 / image.shape[1])
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# sharpening the image
def image_sharpness(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


# writing the selected stone to a new_image
def print_stone(image, i, j, new_image, boarder, min_max):
    max_j = j
    if image[i][j] > 0:
        new_image[i][j] = image[i][j]
        image[i][j] = 0
        if min_max is not None:
            min_max[0] = min(min_max[0], i)
            min_max[1] = max(min_max[1], i)
            min_max[2] = min(min_max[2], j)
            min_max[3] = max(min_max[3], j)
        if 0 < i:
            max_j = max(max_j, print_stone(image, i - 1, j, new_image, boarder, min_max))
        if 0 < j:
            max_j = max(max_j, print_stone(image, i, j - 1, new_image, boarder, min_max))
        if i < image.shape[0] - 1:
            max_j = max(max_j, print_stone(image, i + 1, j, new_image, boarder, min_max))
        if j < image.shape[1] - 1:
            max_j = max(max_j, print_stone(image, i, j + 1, new_image, boarder, min_max))
    else:
        if boarder is not None:
            boarder.append((i, j))
    return max_j


# to select the stones in the image and selectTopStones boolean to selecting the top stones
def top_layer_stones(image, new_image=None, selectTopStones=False):
    dim = image.shape
    if new_image is None:
        new_image = creating_empty_image(dim, image.dtype)
    i = (len(image) // 2) if selectTopStones else 0
    j = 0
    if i > 0:
        boarders = []
        min_maxes = []
        list_of_stones = []
        while j < dim[1]:
            if image[i][j] == 255:
                boarder = []
                min_max = [256, -1, 256, -1]
                list_of_stones.append((i, j))
                j = print_stone(image, i, j, new_image, boarder, min_max)
                boarders.append(boarder)
                min_maxes.append(min_max)
            j += 1
        return list_of_stones, boarders, min_maxes, new_image
    else:
        while j < dim[1]:
            for i in range(dim[0]):
                if image[i][j] == 255:
                    j = print_stone(image, i, j, new_image, None, None)
                    break
            j += 1
        return new_image


# returns each stones embedment in a list
def embedment(image, location_of_stones):
    list_of_embedment = []
    l = len(list_of_embedment)
    for x_min, x_max, y_min, y_max in location_of_stones:
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if 0 < image[i][j] < 255:
                    list_of_embedment.append((y_max - j) / (y_max - y_min))
                    break
            if l != len(list_of_embedment):
                l = len(list_of_embedment)
                break
    return list_of_embedment


# to write the image
def write_image(filename, image):
    if filename[-4:] != ".png":
        filename += ".png"
    if cv2.imwrite(os.getcwd() + "/" + filename, image):
        print(filename, "written successful")
    else:
        print(filename, " written failed")


# function waiting for to close all the tabs opened
def end_program():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = read_image("img1.png")
    img = image_scaling(img)
    show_image("image", img)

    # to process the black line around the stones
    stone_holding_part = image_conversion(img, cv2.COLOR_BGR2GRAY)
    show_image("temp1", stone_holding_part)

    dim = stone_holding_part.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            stone_holding_part[i][j] = 123 if stone_holding_part[i][j] < 30 else 0
    stone_holding_part = image_sharpness(stone_holding_part)
    show_image("temp", stone_holding_part)

    # enhancing the image to capture stones
    img = image_conversion(img, cv2.COLOR_BGR2RGB)
    img = image_contrast_brightness(img)
    img = image_sharpness(img)
    img = image_conversion(img, cv2.COLOR_BGR2GRAY)
    show_image("black", img)

    # converting the image into black and white based on a threshold
    dim = img.shape
    image_bw = creating_empty_image(dim, img.dtype)

    # setting threshold for the black and white image
    threshold = 25
    for i in range(dim[0]):
        for j in range(dim[1]):
            image_bw[i][j] = 255 if img[i][j] > threshold else 0

    # separating the top layer of stones
    all_solid_stones = top_layer_stones(image_bw)
    stones_coordinates, boarders, min_maxes, top_layer = top_layer_stones(all_solid_stones, stone_holding_part, True)
    stones_embedment_percentage = embedment(top_layer, min_maxes)

    for i, j in enumerate(stones_embedment_percentage):
        print(i + 1, ":", j * 100, "%")
    show_image("result", top_layer)

    # writing the processed image to the file
    write_image("result", top_layer)
    end_program()


if __name__ == "__main__":
    main()
