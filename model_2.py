import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import pandas as pd
import os

from operator import itemgetter
lowquality={
"inicountq" : 45,
"inithreshq" : 183,
"inicountid" : 53,
"inithreshid" : 160,
"limitX" : 200,
}

highquality={
    "inicountq" : 1000,
    "inithreshq" : 165,
    "inicountid" : 1250,
    "inithreshid" : 180,
    "limitX" : 600
}


idCount = 40


totalCount = 72
def write_in_xsl(question_results,file) :
    data = pd.DataFrame({
        "Questions": question_results
    })

    def cond(x):
        if x == '':
            return 'background-color: red'
        return None

    data.style.apply(cond)
    data.style.map(cond)

    writer = pd.ExcelWriter('./model_2_results/'+file+'.xlsx', engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Sheet1')

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})

    (max_row, max_col) = data.shape
    worksheet.conditional_format(1, 1, max_row, max_col,
                                 {'type': 'formula',
                                  'criteria': '=B2=""',
                                  'format': format1
                                  }
                                 )

    writer.close()
    print("Created excel file")
def get_code_reults(codes,img,qualitytype):
    rects = []

    for code in codes:
        for i in range(0, 10):
            x_min = code[i][0] - code[i][2]
            y_min = code[i][1] - code[i][2]
            start_point = (x_min, y_min)
            width = 2 * code[i][2]
            height = 2 * code[i][2]
            end_point = (x_min + width, y_min + height)
            rects.append((start_point, end_point))
    coderow = 1
    Answer = 0
    final = []

    for rect in rects:
        # print(rect)
        object = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)

        (thresh, object) = cv2.threshold(object, qualitytype["inithreshid"], 255, cv2.THRESH_BINARY)
        mask = np.zeros(object.shape)
        mask[1:-1, 1:-1] = 1
        count = np.count_nonzero(np.logical_and(object, mask))
        # print(count,Answer)
        if count < qualitytype["inicountid"]:
            result = str(Answer)
            img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
            final.append("code: "+ str(coderow) + ": "+ result)

        Answer = Answer + 1
        if (Answer == 10):
            Answer = 0
            coderow = coderow + 1
    return final

def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated
    # titles. images[0] will be drawn with the title titles[0] if exists You aren't required to understand this
    # function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
def get_perspective(img,contour):
    x, y, w, h = cv2.boundingRect(contour)
    src_points = np.float32(contour)
    # print(src_points.shape)
    dst_points = np.float32([
        [0, 0],
        [0, h-1],
        [w-1, h-1],
        [w-1, 0],
    ])
    # dst_points = np.float32([
    #    [w - 1, 0],
    #    [0, 0],
    #    [0,h-1],
    #    [w-1,h-1],
    #])

    # print(src_points.shape)
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, perspective_matrix, (w, h))
    return warped_image
def detect_allcircles(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    rows = img_gray.shape[0]
    maxRadius = 30

    while maxRadius > 0:
        minRadius = maxRadius-1

        while minRadius < maxRadius:
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT_ALT,
                                       dp=1.5,
                                       minDist=15,
                                       param1=200,
                                       param2=0.9,
                                       minRadius=minRadius,
                                       maxRadius=maxRadius)
            # print(minRadius, maxRadius)
          #  if circles is not None:
            # show_images([img],["title"])


            if circles is not None and (circles.shape[1] == totalCount or circles.shape[1] == totalCount + 1):
               # print(circles.shape[1])


                return circles  # Break out of both loops if the condition is met
            else:
                minRadius -= 1  # Try increasing minRadius

        maxRadius -= 1  # If minRadius loop completes, decrease maxRadius

    return None  # If no suitable combination is found

def read_paper_from_image(path):
    img = cv2.imread(path,1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_edge = cv2.Canny(img_blur,50,100)
    kernel = np.ones((5,5))
    img_dil =cv2.dilate(img_edge,kernel,iterations=2)
    img_ero = cv2.erode(img_dil,kernel,iterations=1)
    img_contours =img.copy()
    contours,hierarchy = cv2.findContours(img_ero,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.09 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    warped = get_perspective(img_contours,approx_polygon)
    return warped
def getPaper(path):
    img = cv2.imread(path)
    img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(img_gray, (5, 5), 0)  # remove noise
    (thresh, edges) = cv2.threshold(blurred_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    edges = cv2.dilate(edges, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    hull = cv2.convexHull(approx_polygon)

    # cv2.drawContours(img, largest_contour, contourIdx=-1, thickness=15, color=(255))

    x, y, w, h = cv2.boundingRect(approx_polygon)
    src_points = np.float32(hull)

    if (len(src_points)) > 4:
        print("Can't detect the box")
        return None

    top_left = [0, 0]
    bot_left = [0, h]
    top_right = [w, 0]
    bot_right = [w, h]

    top_left_i = 0
    bot_left_i = 0
    top_right_i = 0
    bot_right_i = 0

    i = 0
    center = [0, 0]
    while i < len(src_points):
        center += src_points[i][0]
        i = i + 1

    center[0] = center[0] / 4
    center[1] = center[1] / 4
    i = 0
    while i < len(src_points):
        v = src_points[i][0] - center
        if v[0] < 0 and v[1] < 0:
            top_left_i = i
        if v[0] > 0 > v[1]:
            top_right_i = i
        if v[0] > 0 and v[1] > 0:
            bot_right_i = i
        if v[0] < 0 < v[1]:
            bot_left_i = i
        i = i + 1

    dst_points = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0],
    ])

    dst_points[bot_left_i] = bot_left
    dst_points[bot_right_i] = bot_right
    dst_points[top_left_i] = top_left
    dst_points[top_right_i] = top_right

    # print(src_points)
    # print(dst_points)

    # print(src_points.shape)
    # print(dst_points.shape)

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, perspective_matrix, (w, h))
    # show_images([warped_image])

    return warped_image
def get_code_and_questions(circles,qualitytype) :
    circles = circles[0]
    if(circles.shape[0] == totalCount+1 ):
        circles=circles[1:]
    print(circles.shape)
    circles_y = sorted(circles, key=itemgetter(1))

    code_circles = circles_y[0:idCount]
    Question_Circles = circles_y[idCount:]
    limitx = qualitytype["limitX"]
    # Separate elements based on x-axis value
    arr1 = [elem for elem in Question_Circles if elem[0] < limitx]
    arr2 = [elem for elem in Question_Circles if elem[0] >= limitx]

    # Sort both arrays by y-axis value
    arr1.sort(key=lambda x: x[1])
    arr2.sort(key=lambda x: x[1])

    # Concatenate arr2 to arr1
    Question_Circles = arr1 + arr2

    Questions = []
    i = 0
    while i < len(Question_Circles):
        Questions.append(sorted(Question_Circles[i:i + 2], key=itemgetter(0)))
        i = i + 2
    Questions=np.uint16(np.around(Questions))
    codes =[]
    i =0
    while i < len(code_circles):
        codes.append(sorted(code_circles[i:i+10], key=itemgetter(0)))
        i = i+10
    # for code in codes:
    #     for j in range (0,10):
    #         code[j]= {j:code[j]}
    codes = np.uint16(np.around(codes))

    # print(Questions)

    return Questions,codes

def get_questions_results(Questions,img,qualitytype):
    rects = []
    for Question in Questions:
        for i in range(0, 2):
            x_min = Question[i][0] - Question[i][2]
            y_min = Question[i][1] - Question[i][2]
            start_point = (x_min, y_min)
            width = 2 * Question[i][2]
            height = 2 * Question[i][2]
            end_point = (x_min + width, y_min + height)
            rects.append((start_point, end_point))
    QuestionNum = 1
    Answer = 0
    bubbles_count= 0
    final =[]
    for rect in rects:
        object = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)

        (thresh, object) = cv2.threshold(object, qualitytype["inithreshq"], 255, cv2.THRESH_BINARY)
        mask = np.zeros(object.shape)
        mask[1:-1, 1:-1] = 1
        count = np.count_nonzero(np.logical_and(object, mask))

        if count < qualitytype["inicountq"]:
            result = ""
            if (Answer == 0):
                result = 'a'
            elif (Answer == 1):
                result = 'b'

            if(bubbles_count == 0):
                final.append(str(QuestionNum)+" : "+result)
                img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] = 255
                bubbles_count= bubbles_count+1
            else:
                final[-1]= str(QuestionNum)+" : -"
        Answer = Answer + 1
        if (Answer == 2):
            if(bubbles_count==0):
                final.append(str(QuestionNum)+ ": -")
                bubbles_count=0
            Answer = 0
            QuestionNum = QuestionNum + 1
            bubbles_count=0
    return final

def main() :
    files = os.listdir('./model_2_images/')


    jpg_files = [file for file in files if file.lower().endswith(".jpg")]

    for jpg_file in jpg_files:

        file_path = os.path.join("./model_2_images/", jpg_file)
        img = read_paper_from_image(file_path)
        qualitytype=""
        if(img.shape[0]<1000):
            qualitytype=lowquality
        else:
            qualitytype=highquality
        # show_images([img], ["title"])
        circles = detect_allcircles(img)





        Questions, codes = get_code_and_questions(circles,qualitytype=qualitytype)

        question_results = get_questions_results(Questions,img,qualitytype=qualitytype)
        code_results = get_code_reults(codes,img,qualitytype=qualitytype)
        print("code:",code_results)
        write_in_xsl(question_results,jpg_file)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in Questions:
                for j in i:
                    center = (j[0], j[1])
                    radius = j[2]

                    cv2.circle(img, center, radius, (255, 0, 0), 3)
            for i in codes:
                for j in i:
                    center = (j[0], j[1])
                    radius = j[2]
                    cv2.circle(img, center, radius, (0, 0, 255), 3)
            file_path = os.path.join("./model_2_results/", jpg_file)
            cv2.imwrite(file_path, img)


