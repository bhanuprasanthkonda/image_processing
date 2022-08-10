import cv2
import numpy as np
import sys
import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import warnings

warnings.filterwarnings('ignore', '.*', )

sys.setrecursionlimit(100000000)

Debug = False


# reads the image from the filepath
def read_image(filepath):
    return cv2.imread(filepath)


# show the image
def show_image(window_name, image):
    if Debug:
        cv2.imshow(window_name, image)


# returns a converted image
def image_conversion(image, formate):
    return cv2.cvtColor(image, formate)


# returns a new empty image
def creating_empty_image(dim, dtype):
    return np.zeros(dim, dtype)


# setting the contras and brightness of the image
def image_contrast_brightness(image, alpha=2.95, beta=-10):
    new_image = creating_empty_image(image.shape, image.dtype)
    # alpha = 2.95
    # beta = -10
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y, x, 0] = np.clip(alpha * image[y, x, 0] + beta, 0, 255)
    return new_image


# scaling the image to (350,) or (0,2000) dimensions
def image_scaling(image):
    scale_percent = min(23800 / image.shape[0], 200000 / image.shape[1])
    # scale_percent = 23800 / image.shape[0]
    # scale_percent = 200000 / image.shape[1]
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, fx=3, fy=4, interpolation=cv2.INTER_AREA)


# sharpening the image
def image_sharpness(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-2, kernel=kernel)
    return image_sharp


# area of the grey location
def grey_area(image, new_image, x, y, delete=False):
    result = 0
    if delete:
        image = new_image
    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and 0 < image[x][y] < 255:
        temp = image[x][y]
        image[x][y] = 0
        new_image[x][y] = 0
        result = 1 + grey_area(image, new_image, x, y + 1, delete) + grey_area(image, new_image, x, y - 1, delete) \
                 + grey_area(image, new_image, x + 1, y, delete) + grey_area(image, new_image, x - 1, y, delete)
        if not delete:
            new_image[x][y] = temp
    return result


# removes tiny grey parts
def remove_tiny_parts(image):
    new_image = image.copy()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if 0 < image[x][y] < 255:
                result = grey_area(image, new_image, x, y)
                if result <= 0.001 * image.shape[0] * image.shape[1]:
                    grey_area(image, new_image, x, y, True)
    return new_image


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
                min_max = [float('inf'), -1, float('inf'), -1]
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


def overlap(img, toplayer):
    # print(img.shape, toplayer.shape)
    dim = img.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if toplayer[i][j] == 255:
                img[i][j] = [255, 255, 255]
    show_image("overlap", img)
    write_image("overlap", img)
    return img


# returns each stones embedment in a list
def embedment(image, location_of_stones):
    list_of_embedment = []
    lst = []
    initial_len = len(list_of_embedment)
    for x_min, x_max, y_min, y_max in location_of_stones:
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if 0 < image[i][j] < 255:
                    lst.append((j, i))
                    list_of_embedment.append((x_max - i) / (x_max - x_min))
                    break
            if initial_len != len(list_of_embedment):
                initial_len = len(list_of_embedment)
                break
    return list_of_embedment, lst


# put text on the image
def write_on_image(image, pos, text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.45
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    return cv2.putText(image, str(text), pos, font, fontScale, color, thickness, cv2.LINE_AA)


# to write the image
def write_image(filename, image):
    if filename[-4:] != ".png":
        filename += ".png"
    wr = cv2.imwrite(os.getcwd() + "/" + filename, image)
    if Debug and wr:
        print(filename, "written successful")
    elif Debug:
        print(filename, " written failed")


# function waiting for to close all the tabs opened
def end_program():
    if Debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(threshold=20.0, alpha=2.95, beta=-10.0, img="", shouldBreak=False):
    # print(img)
    if img is None:
        img = "img1.png"
    img = read_image(img)
    # print(img[0][0])
    img = image_scaling(img)
    img_copy = img.copy()
    # bm, gm, rm = float("inf"), float("inf"), float("inf")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i][j]
            if r >= 66 and g >= 135 and b >= 155:
                img[i][j] = [133, 57, 10]
            elif r <= 35 and g <= 35 and b <= 35:
                img[i][j] = [0, 0, 0]
    #         elif i == 0:
    #             bm = min(bm, b)
    #             gm = min(gm, g)
    #             rm = min(rm, r)
    # print(bm, gm, rm)
    show_image("image", img)
    # print(img.shape)
    # end_program()
    # return

    # to process the black line around the stones
    grey_scale_img = image_conversion(img, cv2.COLOR_BGR2GRAY)
    show_image("grey_scale_img", grey_scale_img)
    # progress("*" * 2 + " 10% Complete")
    # end_program()
    # return
    dim = grey_scale_img.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            grey_scale_img[i][j] = 123 if grey_scale_img[i][j] < 23 else 0
    stone_holding_part = image_sharpness(grey_scale_img)
    stone_holding_part = remove_tiny_parts(stone_holding_part)
    show_image("stone_holding_part", stone_holding_part)
    # end_program()
    # return
    # progress("**" * 2 + " 20% Complete")

    # enhancing the image to capture stones
    img = image_conversion(img, cv2.COLOR_BGR2RGB)
    img = image_contrast_brightness(img, alpha, beta)
    img = image_sharpness(img)
    img = image_conversion(img, cv2.COLOR_BGR2GRAY)
    show_image("black", img)
    # end_program()
    # return
    # progress("***" * 2 + " 30% Complete")

    # converting the image into black and white based on a threshold
    dim = img.shape
    image_bw = creating_empty_image(dim, img.dtype)

    # setting threshold for the black and white image
    # threshold = 23
    # print("threshold",threshold)
    for i in range(dim[0]):
        for j in range(dim[1]):
            image_bw[i][j] = 255 if img[i][j] > threshold else 0
    show_image("image_bw", image_bw)
    end_program()
    if shouldBreak:
        write_image("image_bw", image_bw)
        # print("complete")
        return image_bw

    # progress("****" * 2 + " 40% Complete")

    # separating the top layer of stones
    all_solid_stones = top_layer_stones(image_bw)
    show_image("all_solid_stones", all_solid_stones)
    # end_program()
    # return
    # progress("*****" * 2 + " 50% Complete")
    stones_coordinates, boarders, min_maxes, top_layer = top_layer_stones(all_solid_stones, stone_holding_part, True)
    # progress("*******" * 2 + " 70% Complete")
    stones_embedment_percentage, lst = embedment(top_layer, min_maxes)
    overlap_img = overlap(img_copy, top_layer).copy()
    # progress("********" * 2 + " 80% Complete")
    top_layer = image_conversion(top_layer, cv2.COLOR_GRAY2RGB)
    # progress("*********" * 2 + " 90% Complete")
    report = []
    for i, j in enumerate(stones_embedment_percentage):
        x, y = 0, 0
        for m, n in boarders[i]:
            x += m
            y += n
        x //= len(boarders[i])
        y //= len(boarders[i])
        j *= 100
        top_layer = write_on_image(top_layer, (y - 20, x), str(j)[:5] + "%")
        overlap_img = write_on_image(overlap_img, (y - 20, x), str(j)[:5] + "%")
        # print(i + 1, ":", j, "%")
        report.append(str(i + 1) + " : " + str(j)[:5] + "%")
    show_image("result", top_layer)
    # progress("**********" * 2 + " 100% Complete")

    # writing the processed image to the file
    write_image("result", top_layer)
    write_image("overlap", overlap_img)
    end_program()
    return overlap_img, report


# def dummy_photo():
#     from random import randint
#     img = read_image("img1.png")
#     img = creating_empty_image(img.shape, img.dtype)
#     print(img.shape)
#     dimx, dimy, dimz = list(img.shape)
#     for i in range(len(img) // 2):
#         for j in range(len(img[0])):
#             img[i][j] = [0, 0, 200]
#     img = image_conversion(img, cv2.COLOR_BGR2RGB)
#     lst = []
#     count = 0
#     for i in range(11):
#         while count == len(lst):
#             x, y = img.shape[0] // 2 - randint(5, 20), randint(0, img.shape[1])
#             l, w = randint(10, 40), randint(10, 40)
#             lst.append((x, y))
#             for m in range(l):
#                 for n in range(w):
#                     if 0 <= x + m < dimx and 0 <= y + n < dimy:
#                         img[x + m][y + n] = [179, 136, 0]
#         count += 1
#     for i in range(10):
#         while count == len(lst):
#             x, y = randint(img.shape[0] // 2 + 10, img.shape[0]), randint(0, img.shape[1])
#             l, w = randint(10, 60), randint(10, 60)
#             lst.append((x, y))
#             for m in range(l):
#                 for n in range(w):
#                     if 0 <= x + m < dimx and 0 <= y + n < dimy:
#                         img[x + m][y + n] = [179, 136, 0]
#         count += 1
#     show_image("kn", img)
#     print(img.shape)
#     # end_program()
#     return img


def GUI():
    root = Tk()
    res = None
    label2 = None
    try:
        root.title("Image_Processing")
        root.geometry("775x700")
        text = StringVar()
        text.set("Please select the file using the browse option")
        Label(root, text="                                     ", justify="center").grid(row=0, column=0)
        Label(root, text="                                     ", justify="center").grid(row=0, column=1)
        Label(root, text="                                     ", justify="center").grid(row=0, column=2)
        Label(root, text="                                     ", justify="center").grid(row=1, column=0)

        L1 = Label(root, textvariable=text, justify="center")
        L1.grid(row=1, columnspan=5)
        # root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(1, weight=1)  # making the input to the center
        label1 = None
        threshold = 23
        inptext = None

        def open_file():
            global label1
            file = filedialog.askopenfile(mode='r')
            filepath = ""
            if file:
                filepath = os.path.abspath(file.name)
                text.set("Selected file: " + str(filepath))
                # L1.grid(row=0, column=0)
                filepath = str(filepath)
                image1 = Image.open(filepath)
                scale_percent = 2000 / image1.width
                image1 = image1.resize(
                    (int(image1.width * scale_percent * 0.35), int(0.35 * image1.height * scale_percent)),
                    Image.ANTIALIAS)
                # print(image1.width, image1.height)
                test = ImageTk.PhotoImage(image1)
                try:
                    label1.configure(image=test)
                    label1.image = test
                except:
                    label1 = Label(image=test, justify="center")
                    label1.image = test
                label1.grid(row=3, column=0, columnspan=5)

            def resultImg():
                global Debug
                temp_Debug = Debug
                Debug = True
                show_image("Result (Press any key to close)", final_img)
                end_program()
                Debug = temp_Debug

            # def GUI_ProgressBar(text):
            #     return
            #     Label(root, text=text, font='Aerial 12', justify="left").grid(row=6, column=0)
            def Analyze():
                global res
                global label2
                global final_img
                threshold = float(inptext_threshold.get().strip())
                # print(threshold)
                alpha = float(inptext_intensity.get().strip())
                beta = float(inptext_reduction.get().strip())
                final_img, report = main(threshold, alpha, beta, filepath, False)
                f_img = Image.open("overlap.png")
                scale_percent = 2000 / f_img.width
                image2 = f_img.resize(
                    (int(f_img.width * scale_percent * 0.35), int(0.35 * f_img.height * scale_percent)),
                    Image.ANTIALIAS)
                f_test = ImageTk.PhotoImage(image2)
                try:
                    label2.destroy()
                    label2 = Label(image=f_test, justify="center")
                    label2.configure(image=f_test)
                    label2.image = f_test
                except:
                    label2 = Label(image=f_test, justify="center")
                    label2.image = f_test
                label2.grid(row=12, column=0, columnspan=5)
                stringreport = "\n".join(report)
                string = StringVar()
                string.set(stringreport)
                try:
                    res.destroy()
                except:
                    pass
                res = Label(root, textvariable=string, font='Aerial 12', justify="left")
                res.grid(row=13, column=1)
                root.grid_columnconfigure(13, weight=1)
                Button(root, text="Result", command=resultImg).grid(row=11, column=1)

            def setThreshold():
                global res
                global label2
                global final_img
                threshold = float(inptext_threshold.get().strip())
                # print(threshold)
                alpha = float(inptext_intensity.get().strip())
                beta = float(inptext_reduction.get().strip())
                f_img = main(threshold, alpha, beta, filepath, True)
                f_img = Image.open("image_bw.png")
                scale_percent = 2000 / f_img.width
                image2 = f_img.resize(
                    (int(f_img.width * scale_percent * 0.35), int(0.35 * f_img.height * scale_percent)),
                    Image.ANTIALIAS)
                f_test = ImageTk.PhotoImage(image2)
                try:
                    label2.destroy()
                    label2 = Label(image=f_test, justify="center")
                    label2.configure(image=f_test)
                    label2.image = f_test
                except:
                    label2 = Label(image=f_test, justify="center")
                    label2.image = f_test
                label2.grid(row=9, column=0, columnspan=5)
                # stringreport = "\n".join(report)
                # string = StringVar()
                # string.set(stringreport)
                # try:
                #     res.destroy()
                # except:
                #     pass
                # res = Label(root, textvariable=string, font='Aerial 12', justify="left")
                # res.grid(row=9, column=1)
                # root.grid_columnconfigure(11, weight=1)
                # Button(root, text="Result", command=resultImg).grid(row=6, column=1)

            def Preview():
                global Debug
                temp_Debug = Debug
                Debug = True
                show_image("Preview (Press any key to close)", read_image(filepath))
                end_program()
                Debug = temp_Debug

            if filepath:
                Button(root, text="Preview", command=Preview).grid(row=4, column=1)
                Button(root, text="Start Analyse", command=Analyze).grid(row=10, column=1)
                Button(root, text="Load Image", command=setThreshold).grid(row=8, column=1)

                Label(root, text="Threshold:").grid(row=5, column=1)
                # inptext_threshold = StringVar()
                inptext_threshold = Entry(root)
                inptext_threshold.grid(row=5, column=2)
                inptext_threshold.focus()
                inptext_threshold.insert(0, "20")

                Label(root, text="Intensity:").grid(row=6, column=1)
                # inptext_text = StringVar()
                inptext_intensity = Entry(root)
                inptext_intensity.grid(row=6, column=2)
                inptext_intensity.focus()
                inptext_intensity.insert(0, "2.95")

                Label(root, text="Reduction:").grid(row=7, column=1)
                # inptext_ = StringVar()
                inptext_reduction = Entry(root)
                inptext_reduction.grid(row=7, column=2)
                inptext_reduction.focus()
                inptext_reduction.insert(0, "-10")

        Button(root, text="Browse", command=open_file).grid(row=2, column=1)

        root.mainloop()
    except:
        pass
    finally:
        # root.destroy()
        sys.exit()


if __name__ == "__main__":
    # img = dummy_photo()
    # write_image("img2.png", img)
    # show_image("dummy", img)
    GUI()
    # main("IMG_6228.png")
