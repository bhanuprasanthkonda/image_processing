import cv2
import numpy as np
import sys
import os
from tkinter import *
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
from datetime import datetime

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
def image_contrast_brightness(image, alpha=3.0, beta=-10):
    new_image = creating_empty_image(image.shape, image.dtype)
    # alpha = 3.0
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
    # print(location_of_stones)
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


def draw_on_image(image, min_max):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.45
    # Blue color in BGR
    color = (255, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method

    for x_min, x_max, y_min, y_max in min_max:
        start_point = (y_min, x_min)

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (y_max, x_max)

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


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


def main(img, threshold=[20.0, 27.0], alpha=3.0, beta=-10.0, stone_holding_part="",
         img_copy="", stones_coordinates=None, boarders=None, min_maxes=None, top_layer=None):
    if isinstance(img, str):
        img = read_image(img)
        img = image_scaling(img)
        img_copy = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b, g, r = img[i][j]
                if r >= 66 and g >= 135 and b >= 155:
                    img[i][j] = [133, 57, 10]
                elif r <= 35 and g <= 35 and b <= 35:
                    img[i][j] = [0, 0, 0]
        show_image("image", img)

        # to process the black line around the stones
        grey_scale_img = image_conversion(img, cv2.COLOR_BGR2GRAY)
        show_image("grey_scale_img", grey_scale_img)

        dim = grey_scale_img.shape
        stone_holding_part_threshold = threshold[1]
        for i in range(dim[0]):
            for j in range(dim[1]):
                grey_scale_img[i][j] = 123 if grey_scale_img[i][j] < stone_holding_part_threshold else 0
        stone_holding_part = image_sharpness(grey_scale_img)
        stone_holding_part = remove_tiny_parts(stone_holding_part)
        show_image("stone_holding_part", stone_holding_part)

        # enhancing the image to capture stones
        img = image_conversion(img, cv2.COLOR_BGR2RGB)
        img = image_contrast_brightness(img, alpha, int(beta))
        img = image_sharpness(img)
        img = image_conversion(img, cv2.COLOR_BGR2GRAY)
        show_image("black", img)

        # converting the image into black and white based on a threshold
        dim = img.shape
        image_bw = creating_empty_image(dim, img.dtype)

        # setting threshold for the black and white image
        # threshold = 23
        for i in range(dim[0]):
            for j in range(dim[1]):
                image_bw[i][j] = 255 if img[i][j] > threshold[0] else 0
        show_image("image_bw", image_bw)

        # if shouldBreak:
        write_image("image_bw", image_bw)

        # separating the top layer of stones
        all_solid_stones = top_layer_stones(image_bw)
        show_image("all_solid_stones", all_solid_stones)

        stones_coordinates, boarders, min_maxes, top_layer = top_layer_stones(all_solid_stones, stone_holding_part,
                                                                              True)
        preview_img = creating_empty_image(dim, img.dtype)
        for i in range(dim[0]):
            for j in range(dim[1]):
                preview_img[i][j] = 255 if top_layer[i][j] == 255 else (123 if grey_scale_img[i][j] == 123 else 0)
        preview_img = draw_on_image(preview_img, min_maxes)
        show_image("preview_img", preview_img)
        show_image("top_layer_2", top_layer)
        write_image("prev_image_bw", preview_img)
        end_program()
        return image_bw, stone_holding_part, img_copy, stones_coordinates, boarders, min_maxes, top_layer

    # stones_coordinates, boarders, min_maxes, top_layer
    stones_embedment_percentage, lst = embedment(top_layer, min_maxes)
    overlap_img = overlap(img_copy, top_layer).copy()
    top_layer = image_conversion(top_layer, cv2.COLOR_GRAY2RGB)
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
        report.append(str(i + 1) + " : " + str(j)[:5] + "%")
    show_image("result", top_layer)

    # writing the processed image to the file
    write_image("result", top_layer)
    write_image("overlap", overlap_img)
    end_program()
    return overlap_img, report


def GUI():
    column = 0
    width, height = 725, 775
    main_root = Tk()
    res = None
    label2 = None
    main_frame = Frame(main_root)
    main_frame.pack(fill=BOTH, expand=1)

    my_canvas = Canvas(main_frame, scrollregion=(0, 0, 1075, 775))
    my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

    my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    my_scrollbar.pack(side=RIGHT, fill=Y)

    my_canvas.configure(yscrollcommand=my_scrollbar.set)
    my_canvas.configure(scrollregion=my_canvas.bbox("all"))

    root = Frame(my_canvas)
    my_canvas.create_window((0, 0), window=root, anchor="nw")
    main_root.bind_all('<MouseWheel>', lambda event: my_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
    try:
        main_root.title("Image_Processing")
        main_root.geometry(str(width) + "x" + str(height))
        text = StringVar()
        text.set(
            "                                                          Please select the file using the browse option                                                 ")

        L1 = Label(root, textvariable=text, justify="center")
        L1.grid(row=1, column=column, columnspan=3)

        root.grid_columnconfigure(column, weight=1)  # making the input to the center
        label1 = None
        loaded_img = ""
        stone_holding_part = ""
        img_copy = ""
        stones_coordinates = []
        boarders = []
        min_maxes = []
        top_layer = None

        def open_file():
            global label1, preview_button, inptext_threshold_text, inptext_threshold, inptext_intensity_text, res,\
        inptext_intensity, inptext_reduction_text, inptext_reduction, load_button, label2, analyse_button, result_button
            file = filedialog.askopenfile(mode='r')
            filepath = ""
            if file:
                filepath = os.path.abspath(file.name)
                if os.stat(filepath).st_size / 1024 > 1024 or filepath[-4:] != ".png":
                    text.set(
                        "                  Please select the file using the browse option (file size should be less than 1MB and in png format)                     ")
                    filepath = ""
                    try:
                        label1.destroy()
                    except:
                        pass
                    try:
                        preview_button.grid_remove()
                    except:
                        pass
                    try:
                        inptext_threshold_text.grid_remove()
                    except:
                        pass
                    try:
                        inptext_threshold.grid_remove()
                    except:
                        pass
                    try:
                        inptext_intensity_text.grid_remove()
                    except:
                        pass
                    try:
                        inptext_intensity.grid_remove()
                    except:
                        pass
                    try:
                        inptext_reduction_text.grid_remove()
                    except:
                        pass
                    try:
                        inptext_reduction.grid_remove()
                    except:
                        pass
                    try:
                        load_button.grid_remove()
                    except:
                        pass
                    try:
                        lastloaded_text_label.grid_remove()
                    except:
                        pass
                    try:
                        label2.destroy()
                        analyse_button.grid_remove()
                        result_button.grid_remove()
                    except:
                        pass
                    try:
                        res.grid_remove()
                    except:
                        pass
                    return
                text.set("Selected file: " + str(filepath))
                filepath = str(filepath)
                image1 = Image.open(filepath)
                scale_percent = 2000 / image1.width
                image1 = image1.resize(
                    (int(image1.width * scale_percent * 0.35), int(0.35 * image1.height * scale_percent)),
                    Image.ANTIALIAS)
                test = ImageTk.PhotoImage(image1)
                try:
                    label1.destroy()
                    label1.configure(root, image=test)
                    label1.image = test
                except:
                    label1 = Label(root, image=test, justify="center")
                    label1.image = test
                label1.grid(row=3, column=column, columnspan=3)

            def resultImg():
                global Debug
                temp_Debug = Debug
                Debug = True
                show_image("Result (Press any key to close)", final_img)
                end_program()
                Debug = temp_Debug

            def Analyze():
                global res
                global label2
                global final_img
                global loaded_img
                global stone_holding_part
                global img_copy, stones_coordinates, boarders, min_maxes, top_layer, result_button, res
                threshold = list(map(float, inptext_threshold.get().strip().split(",")))
                alpha = float(inptext_intensity.get().strip())
                beta = float(inptext_reduction.get().strip())
                if isinstance(loaded_img, str):
                    loaded_img, stone_holding_part, img_copy, stones_coordinates, boarders, min_maxes, top_layer = main(
                        filepath, threshold, alpha, beta)
                final_img, report = main(loaded_img, threshold, alpha, beta, stone_holding_part, img_copy,
                                         stones_coordinates, boarders, min_maxes, top_layer)
                f_img = Image.open("overlap.png")
                scale_percent = 2000 / f_img.width
                image2 = f_img.resize(
                    (int(f_img.width * scale_percent * 0.35), int(0.35 * f_img.height * scale_percent)),
                    Image.ANTIALIAS)
                f_test = ImageTk.PhotoImage(image2)
                try:
                    label2.destroy()
                    label2 = Label(root, image=f_test, justify="center")
                    label2.configure(image=f_test)
                    label2.image = f_test
                except:
                    label2 = Label(root, image=f_test, justify="center")
                    label2.image = f_test
                label2.grid(row=12, column=column, columnspan=3)
                # label2.pack()
                report.insert(0, "Result(Copied to Clipboard) \n(Generated at " + datetime.now().strftime(
                    "%d/%m/%Y %H:%M:%S") + ")")
                stringreport = "\n".join(report)
                try:
                    res.destroy()
                except:
                    pass

                res = Text(root, height=12)
                res.insert(END, stringreport, "result")
                res.configure(state=DISABLED)
                res.tag_config("result", justify='center')
                res.grid(row=13, column=column, columnspan=3)
                root.clipboard_clear()
                root.clipboard_append(stringreport)
                root.grid_columnconfigure(0, weight=1)
                result_button = Button(root, text="Result", command=resultImg)
                result_button.grid(row=11, column=column, columnspan=3)

            def setThreshold():
                global res
                global label2
                global final_img
                global loaded_img
                global stone_holding_part
                global img_copy
                global stones_coordinates, boarders, min_maxes, top_layer, analyse_button
                lastloaded_text.set("Loading")
                threshold = list(map(float, inptext_threshold.get().strip().split(",")))
                alpha = float(inptext_intensity.get().strip())
                beta = float(inptext_reduction.get().strip())
                loaded_img, stone_holding_part, img_copy, stones_coordinates, boarders, min_maxes, top_layer = main(
                    filepath, threshold, alpha, beta)
                f_img = Image.open("prev_image_bw.png")
                scale_percent = 2000 / f_img.width
                image2 = f_img.resize(
                    (int(f_img.width * scale_percent * 0.35), int(0.35 * f_img.height * scale_percent)),
                    Image.ANTIALIAS)
                f_test = ImageTk.PhotoImage(image2)
                try:
                    label2.destroy()
                    label2 = Label(root, image=f_test, justify="center")
                    label2.configure(image=f_test)
                    label2.image = f_test
                except:
                    label2 = Label(root, image=f_test, justify="center")
                    label2.image = f_test
                label2.grid(row=9, column=column, columnspan=3)
                lastloaded_text.set(str(datetime.now().strftime("Last processed: %d/%m/%Y %H:%M:%S")))
                analyse_button = Button(root, text="Start Analyse", command=Analyze)
                analyse_button.grid(row=10, column=column, columnspan=3)

            def Preview():
                global Debug
                temp_Debug = Debug
                Debug = True
                show_image("Preview (Press any key to close)", read_image(filepath))
                end_program()
                Debug = temp_Debug

            if filepath:
                preview_button = Button(root, text="Preview", command=Preview)
                preview_button.grid(row=4, column=column, columnspan=3)

                inptext_threshold_text = Label(root, text="Threshold(stone,adhesive):")
                inptext_threshold_text.grid(row=5, column=column, columnspan=2)
                inptext_threshold = Entry(root)
                inptext_threshold.grid(row=5, column=column + 1)
                inptext_threshold.focus()
                inptext_threshold.insert(0, "20,30")

                inptext_intensity_text = Label(root, text="Intensity:")
                inptext_intensity_text.grid(row=6, column=column, columnspan=2)
                inptext_intensity = Entry(root)
                inptext_intensity.grid(row=6, column=column + 1)
                inptext_intensity.focus()
                inptext_intensity.insert(0, "3.0")

                inptext_reduction_text = Label(root, text="Reduction:")
                inptext_reduction_text.grid(row=7, column=column, columnspan=2)
                inptext_reduction = Entry(root)
                inptext_reduction.grid(row=7, column=column + 1)
                inptext_reduction.focus()
                inptext_reduction.insert(0, "-10")

                load_button = Button(root, text="Load Image", command=setThreshold)
                load_button.grid(row=8, column=column, columnspan=3)

        Button(root, text="Browse", command=open_file).grid(row=2, column=column, columnspan=3)
        lastloaded_text = StringVar()
        lastloaded_text_label = Label(root, textvariable=lastloaded_text)
        lastloaded_text_label.grid(row=8, column=column + 1, columnspan=3)

        main_root.mainloop()
    except:
        pass
    finally:
        sys.exit()


if __name__ == "__main__":
    GUI()
