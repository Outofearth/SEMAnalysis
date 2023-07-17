import cv2 as cv
import numpy as np
import pyautogui as pg
from matplotlib import pyplot as plt
import time


# 与V1版相比，V2版实现对话框选择粒径预估范围，从而循环选择给定粒径范围参数  20221119
# 更新了hough参数  20221124
# 更新自动截取比例尺像素数 20230615

def hough_lines(img_roi):
    # image = cv.imread(ROI_PIC)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # img_roi = cv.selectROI('ROI',image)
    # print('Select width is {} pix'.format(img_roi[2]))
    # img_roi = image[img_roi[1]:img_roi[1] + img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
    edges = cv.Canny(img_roi, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        lines=0,
        minLineLength=5,  # Min allowed length of line
        maxLineGap=100  # Max allowed gap between line for joining them
    )
    for x1, y1, x2, y2 in lines[0]:
        cv.line(img_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow('Check scale length', img_roi)
    print('Auto detect width is {} pix'.format(x2 - x1))
    k = cv.waitKey(0)
    cv.destroyWindow('Check scale length')
    if k == 27:
        cv.destroyAllWindows()
        exit()
    return x2-x1+4

def hough_circles(pic_name):  # 第一个参数是图形文件
    src = cv.imread(pic_name)
    if src is None:
        pg.alert('请检查文件是否存在或文件名是否正确！')
        exit()
    cv.namedWindow(pic_name[:5], 1)  # cv.WINDOW_KEEPRATIO
    img_roi = cv.selectROI(pic_name[:5], src)
    # pix = img_roi[2]
    # Crop image
    # img[y1:y2,x1:x2]进行ROI截取,注意坐标顺序
    img_roi = src[img_roi[1]:img_roi[1] + img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
    pix = hough_lines(img_roi)
    cv.destroyWindow(pic_name[:5])
    scaleInput = int(pg.prompt("Please input scale value", default='500'))
    # 显示截取的比例尺区域图像
    # cv.namedWindow(pic_name[:5] + 'ROI',1)
    # cv.imshow(pic_name[:5] + 'ROI', img_roi)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
        exit()
    sc_value = scaleInput / pix

    # dst = cv.medianBlur(src, 9)
    dst = cv.pyrMeanShiftFiltering(src, 5, 100)
    dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    # ret, dst = cv.threshold(dst, 50, 200, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow('dst binary',dst)
    # cv.waitKey(0)
    # dst_roi = dst
    dst_roi = dst[0:dst.shape[0] - 120, 0:dst.shape[1]]
    if scaleInput == 500:  # 500纳米比例尺,
        # es_size = pg.confirm("请选择最接近预估颗粒尺寸，单位nm", buttons=['150','200','300','400'])
        # print(es_size)
        circles = adjust_hough_para500(dst_roi)
        circles = np.uint16(np.around(circles))[0]
    if scaleInput == 1000:  # 1000纳米比例尺,
        circles = adjust_hough_para1000(dst_roi)
        circles = np.uint16(np.around(circles))[0]
    if scaleInput == 2000:  # 1000纳米比例尺,
        circles = adjust_hough_para2000(dst_roi)
        circles = np.uint16(np.around(circles))[0]
    return circles, sc_value


def adjust_hough_para500(image_ROI):
    es_size = pg.confirm("当前是500nm比例尺，请选择最接近预估颗粒尺寸，单位nm", buttons=['150', '200', '250', '350'])
    if int(es_size) == 200:
        print('200 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 60,  # 检测到的圆的圆心之间的最小距离
            'param1': 50,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 50,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 50,  # 圆半径的最小值
            'maxRadius': 80  # 圆半径的最大值
        }
    if int(es_size) == 150:
        print('150 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 80,  # 检测到的圆的圆心之间的最小距离
            'param1': 65,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 55,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 34,  # 圆半径的最小值
            'maxRadius': 59  # 圆半径的最大值63
        }
    if int(es_size) == 250:
        print('300 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 60,  # 检测到的圆的圆心之间的最小距离
            'param1': 50,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 50,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 70,  # 圆半径的最小值
            'maxRadius': 130  # 圆半径的最大值
        }
    if int(es_size) == 350:
        print('400 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 80,  # 检测到的圆的圆心之间的最小距离
            'param1': 50,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 50,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 100,  # 圆半径的最小值
            'maxRadius': 200  # 圆半径的最大值
        }
    circles = cv.HoughCircles(**default_houghcirclePara)
    return circles


def adjust_hough_para1000(image_ROI):
    es_size = pg.confirm("当前是1000nm比例尺，请选择最接近预估颗粒尺寸，单位nm", buttons=['300', '350', '400', '450'])

    if int(es_size) == 300:
        print('1150 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 80,  # 检测到的圆的圆心之间的最小距离
            'param1': 200,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 50,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 34,  # 圆半径的最小值
            'maxRadius': 53  # 圆半径的最大值
        }
    if int(es_size) == 350:
        print('1200 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 60,  # 检测到的圆的圆心之间的最小距离
            'param1': 60,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 60,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 30,  # 圆半径的最小值
            'maxRadius': 70  # 圆半径的最大值
        }
    if int(es_size) == 400:
        print('1200 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 70,  # 检测到的圆的圆心之间的最小距离
            'param1': 60,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 60,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 50,  # 圆半径的最小值
            'maxRadius': 75  # 圆半径的最大值
        }
    circles = cv.HoughCircles(**default_houghcirclePara)
    return circles


def adjust_hough_para2000(image_ROI):
    es_size = pg.confirm("当前是2000nm比例尺，请选择最接近预估颗粒尺寸，单位nm", buttons=['150', '200', '250', '300'])
    if int(es_size) == 150:
        print('1200 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 15,  # 检测到的圆的圆心之间的最小距离
            'param1': 100,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 18,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 10,  # 圆半径的最小值
            'maxRadius': 18  # 圆半径的最大值
        }
    if int(es_size) == 200:
        print('1150 selected')
        default_houghcirclePara = {  # 初始参数
            'image': image_ROI,  # 图像文件
            'method': cv.HOUGH_GRADIENT,  # houfu 梯度法, 目前仅有此一项
            'dp': 2,  # 检测圆心的累加器图像的分辨率于输入图像之比的倒数，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
            'minDist': 15,  # 检测到的圆的圆心之间的最小距离.
            'param1': 100,  # 传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
            'param2': 18,  # 在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
            'minRadius': 10,  # 圆半径的最小值
            'maxRadius': 25  # 圆半径的最大值
        }
    circles = cv.HoughCircles(**default_houghcirclePara)
    return circles


def draw_cal(pic, circles, kv):
    src = cv.imread(pic)
    j = 0
    rlist = []
    vb = 0

    for i in circles:
        # draw the outer circle
        cv.circle(src, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # draw the center of the circle
        cv.circle(src, (i[0], i[1]), 2, (0, 0, 0), 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        j = j + 1
        tstr_n = str(j)
        tstr = str(round((i[2] * 2 * kv), 1))
        cv.putText(src, tstr_n, (i[0], i[1] + 20), font, 0.7, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, tstr, (i[0], i[1] + 40), font, 0.5, (100, 25, 100), 1, lineType=cv.LINE_AA)
    cv.namedWindow(pic[:5] + '-' + str(j), 1)
    cv.moveWindow(pic[:5] + '-' + str(j),50,0) # Keep piciture position on left-upper
    cv.imshow(pic[:5] + '-' + str(j), src)
    k = cv.waitKey(0)

    if k == ord('E' or 'e'):
        cv.destroyWindow(pic[:5] + '-' + str(j))
        dv = pg.prompt('Please input delete number! PRESS ENTER')
        circles = np.delete(circles, int(dv) - 1, axis=0)
        draw_cal(pic, circles, kv)

    elif k == ord('S' or 's'):
        for i in circles:
            rlist = np.append(rlist, i[2])
        # print(rlist)
        # 相对标准偏差/算数平均值，计算公式参考https://zhidao.baidu.com/question/103069634.html
        rsd = np.std(rlist, ddof=1) / np.mean(rlist)
        mn = 'RSD/PDI %: ' + str(round(rsd * 100, 2)) + "%"
        print("相对标准偏差百分比：%.2f" % (rsd * 100) + str(" %"))

        totalNumber = np.size(rlist)
        tn = 'Particle number: ' + str(totalNumber) + " qty"
        print("Total detected circle number:%d" % (np.size(rlist)))

        # meanNumber = np.median(rlist) * 2 * kv
        # mn = 'Mean value: ' + str(round(meanNumber, 1)) + " nm"
        # print("中位数：%.2f nm" % (np.median(rlist) * 2 * kv))

        averageNumber = np.mean(rlist) * 2 * kv
        an = 'Avg value: ' + str(round(averageNumber, 1)) + " nm"
        print("算数平均值:%.2f nm" % (np.mean(rlist) * 2 * kv))

        stdNumber = np.std(rlist, ddof=1) * 2 * kv
        sn = 'Std value: ' + str(round(stdNumber, 1)) + " nm"
        print("标准差:%.2f nm" % (np.std(rlist, ddof=1) * 2 * kv))

        deltaNumber = np.var(rlist, ddof=1) * 2 * kv
        den = 'Div value: ' + str(round(deltaNumber, 1)) + " nm"
        print("方差:%.2f nm" % (np.var(rlist, ddof=1) * 2 * kv))

        minNumber = np.amin(rlist) * 2 * kv
        miin = 'Min value: ' + str(round(minNumber, 1)) + " nm"
        print("Min:%.2f nm" % (np.amin(rlist) * 2 * kv))
        maxNumber = np.amax(rlist) * 2 * kv
        mxn = 'Max value: ' + str(round(maxNumber, 1)) + " nm"
        print("Max:%.2f nm" % (np.amax(rlist) * 2 * kv))
        divNumber = np.ptp(rlist) * 2 * kv
        din = 'Range value: ' + str(round(divNumber, 1)) + " nm"
        print("极差:%.2f nm" % (np.ptp(rlist) * 2 * kv))

        date = time.strftime('%Y%m%d', time.localtime())
        condition_name = pic[:5] + '_P' + str(j) + '_' + date
        # cv.rectangle(src,(x1,y1),(x2,y2),(color),thickness=-1) thickness为负1实体填充
        cv.rectangle(src, (5, src.shape[0] - 78), (src.shape[1] - 1050, src.shape[0] - 5), (0, 0, 0),
                     thickness=-1)  # 添加右下角数据区，背景黑色
        cv.putText(src, condition_name, (10, 1020), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, tn, (10, 1040), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, an, (10, 1059), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, miin, (10, 1079), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, sn, (10, 1098), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, mn, (260, 1040), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, mxn, (260, 1059), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, din, (260, 1079), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)
        cv.putText(src, den, (260, 1098), font, 0.5, (255, 255, 255), 1, lineType=cv.LINE_AA)

        pic_name = pic[:5] + '_PSMReport_P' + str(j) + '_' + date + '.jpg'
        cv.imwrite(pic_name, src)
        cv.namedWindow(pic_name, 1)
        cv.moveWindow(pic_name,50,0) # Keep the picture on left-upper position
        cv.imshow(pic_name, src)
        k = cv.waitKey(0)
        arry = rlist * 2 * kv
        normline(arry, pic_name, round(np.mean(rlist) * 2 * kv, 2), round(np.std(rlist, ddof=1) * 2 * kv, 2),
                 round(rsd * 100, 2), condition_name)
        cv.destroyAllWindows()

    elif k == ord('n' or 'N'):
        cv.destroyAllWindows()
    elif k == 27:
        cv.destroyAllWindows()
        exit()


def normline(arry, pic_name, ave, std, stdp, cdn):
    # array:1维数组；pic_name:保存文件名；ave:数字算数平均值；std：数组标准偏差；stdp：数组相对标准偏差百分比；cdn：文件名
    img = cv.imread(pic_name)
    yd = img.shape[0]
    xd = img.shape[1]
    fig = plt.figure(figsize=(xd / 400, yd / 200), dpi=150, facecolor='blue')
    fig.canvas.set_window_title('Particle Size Distribution by LJ')  # 改变窗体名称，保存图片不显示
    # fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.0)
    mu = arry.mean()
    sigma = arry.std()
    x = np.arange(arry.min(), arry.max(), 0.1)
    y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    plt.plot(x, y)
    # 绘制数据集的直方图
    plt.hist(arry, bins=12, rwidth=0.6, density=True)
    xsc = np.arange(round(arry.min(), 0), round(arry.max(), 0), 5)  # 设置坐标轴大小和分辨率
    plt.xticks(xsc, rotation=90, fontsize=7)  # 坐标文字反向旋转90度，设置字体大小
    title_name = 'Particle size:' + str(ave) + '+/-' + str(std) + ' nm'
    # plt.title('Std%:'+str(stdp)+'%',loc='right',y=0,fontsize=8,color='r')
    plt.title(cdn + '\nParticle distribution analysis\n' + 'PDI/SRD: ' + str(stdp) + ' %', color='r', fontsize=12)
    plt.xlabel(title_name, color='r', fontsize=12)
    plt.ylabel('Probability')
    plt.tight_layout()  # 防止标签被遮挡
    plt.ion()  # 交互显示模式
    pic_name1 = pic_name[:-4] + '_N1.jpg'
    plt.savefig(pic_name1, dpi=150)
    plt.pause(5)  # 搭配plt.ion使用
    # plt.show()
    # time.sleep(5)
    plt.close()


if __name__ == '__main__':
    # hough_lines("1_003.tif")

    try:
        file_name_surfixbumber = pg.prompt('请输入文件名后缀', default='_003.tif')
        start_file_number = int(pg.prompt('Please input start file number!', default='1'))
        pg.press('capslock')
        for i in range(start_file_number, start_file_number + 100, 1):
            pic = str(i) + file_name_surfixbumber
            # pic = str(i) + '_003.tif'
            # print(pic)
            # 打印原始图像尺寸
            # print('%s size  X:%d pix  Y:%d pix'%(pic[:-4],cv.imread(pic).shape[1],cv.imread(pic).shape[0]))
            c1, c2 = hough_circles(pic)
            draw_cal(pic, c1, c2)

            rp = pg.confirm('Do you want to run the next file?', buttons=['Yes', 'Retry', 'No'])
            if rp == 'Retry':
                cv.destroyAllWindows()
                c1, c2 = hough_circles(pic)
                draw_cal(pic, c1, c2)
                rp = pg.confirm('Do you want to run the next file?', buttons=['Yes', 'Retry', 'No'])
            if rp == 'No':
                pg.press('capslock')
                exit()
        pg.press('capslock')
    finally:
        cv.destroyAllWindows()
        exit()
