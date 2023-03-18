import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
THRESH = 30  # ngưỡng này là ngưỡng bổ sung để làm cho đường biên được tách biệt rõ hơn , còn thuật toán canny đã cài sẵn r


class EdegeDetection:
    def __init__(self, img_path):
        images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.images = images
        self.image = cv2.GaussianBlur(images, ksize=(3, 3), sigmaX=0, sigmaY=0)
        self.sobel_img = EdegeDetection.sobel_edge_detection(self)  # phương thức
        self.prewit_img = EdegeDetection.prewit_edgeDetection(self)
        self.canny_img = EdegeDetection.canny_edgeDectection(self)
        self.equa = EdegeDetection.imagePrecesing(self)
        self.histogram = EdegeDetection.histogram_image(self)
        self.CNN_img = self.image

    def imagePrecesing(self):
        equa = cv2.equalizeHist(self.images)
        return equa

    def edgeshow(self):
        titles = ["origin", "sobel", "prewitt", "canny"]
        n = len(titles)
        edges = [self.image, self.sobel_img, self.prewit_img, self.canny_img]
        for i in range(n):
            print(i)
            plt.subplot(1, n, i + 1)
            plt.imshow(edges[i], cmap='gray')
            plt.title(titles[i])
        plt.show()

    def sobel_edge_detection(self):
        x = np.uint8(np.abs(cv2.Sobel(self.image, cv2.CV_64F, dx=1, dy=0, ksize=1)))
        y = np.uint8(np.abs(cv2.Sobel(self.image, cv2.CV_64F, dx=0, dy=1, ksize=1)))
        # sử dụnng scharr ko hiệu quả
        sobel = (x + y) / 2
        sobel[sobel < THRESH] = 0
        sobel[sobel > THRESH] = 255
        return sobel

    def prewit_edgeDetection(self):
        # có 4 mặt nạ và ta nhân 4 mặt nạ vào và lấy trung bình
        kernels, imgs = [0] * 4, []
        print(kernels)
        kernels[0] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernels[1] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernels[2] = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # trắng sang đen của 0
        kernels[3] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # trắng sang đen của 1
        for i in range(len(kernels)):
            # tính corealation
            imgs.append(cv2.filter2D(self.image, ddepth=-1, kernel=kernels[i]))
        img = np.mean(imgs, axis=0)
        img[img < THRESH] = 0
        img[img >= THRESH] = 255
        return img

    def canny_edgeDectection(self):
        img = cv2.Canny(self.image, 30, 170)
        img[img > THRESH] = 255
        img[img < THRESH] = 0
        return img

    def CNN(self):
        # đưa bộ trọng số ở trên vào

        input = Input(shape=[*self.image.shape, 1], batch_size=1, name="input")
        x = conv2D(2, (3, 3), (1, 1), 'same', name='Gxy', activation=relu, use_bias=False)(
            input)  # 2 bộ lọc , kích thuowsc 3x3, stride = (1,1) ,padding = same,
        x = conv2D(1, (1, 1), (1, 1), 'same', name='avg', activation=relu, use_bias=False)(x)
        # (1,1) : tính trung bình 2 lớp trên ,(1,1) 2 : stride
        return Model(input, x, name='CNN_DECTECTION')

    def DNN_edge_detection(self):
        # có thể dùng các conv khác làm mịn ảnh trong model
        img = np.array(self.image)
        img = img.reshape(1, *self.image.shape, 1)  # batch_size =1 và reshape về kích thước ảnh trong mô hình
        model = self.CNN()
        output = model.predict(img)
        result = np.array(output[0, :, :, 0])  #
        result[result < THRESH] = 0
        result[result > THRESH] = 255
        return result

    def histogram_image(self):
        plt.hist(self.equa.ravel(), 256, [0, 256])
        plt.show()

    def hough_line_detection(self):
        dst = self.canny_img  # lấy các đường biên từ pp canny
        dst_color = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, minLineLength=20,
                                 maxLineGap=50)  # trả về danh sách các đt đc phát hiện , mỗi đường thằng đc bieeru diễn bởi 2 điểm
        # tìm đc cả điểm đầu và cuối của đường thẳng, 1 : quét qua tất cả th ,np.pi/180: độ  ( theta quay trong 1 lần phát hiện đường thẳng)
        # 100 : số lượng các điểm trong ko gian houghs xác định là đt ( ngưỡng voting)
        # miniLinelLength : độ dài tối thiểu  xác định là 1 đt
        # tìm đường biên thẳng qua canny và vẽ lên hình luôn
        # maxLineGap : khoảng cách tối đa đc phép giữa các đoạn thẳng > đt duy nhất
        # đơn vị đo bằng pixel
        if linesP is not None:
            for i in range(len(linesP)):
                l = linesP[i][0]  # toạ độ đầu tiên tương ứnng điểm đầu cuối đường thằng
                cv2.line(dst_color, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3,
                         cv2.LINE_AA)  # 255,0,0: vẽ màu đỏ , 3 là độ dày,cv2.LINE_AA: kiểu đường thẳng
        return dst_color

    def lineShow(self):
        showLine = self.hough_line_detection()
        plt.imshow(showLine)
        plt.show()


# Đánh giá xem phương pháp nào tốt
class Evaluation:
    def __init__(self, data_folder):
        # đường dẫn tới ảnh
        self.list_paths = [os.path.join(data_folder, file_name) for file_name in
                           os.listdir(data_folder)]  # nối tên thư mục với tên ảnh (file_name) nằm trong thư mục đó
        # tìm biên theo 3 pp và vẽ lên hist xem biểu cái nào ok nhất

    def plot_histoGram(self):
        hisgram = dict({'sobel': [], "prewit": [], "canny": []})
        for img_path in self.list_paths:
            img_edges = EdegeDetection(img_path)  # khởi tạo đối tượng
            edges = [img_edges.sobel_img, img_edges.prewit_img, img_edges.canny_img]  # chứa tất cả các pp
            for i, k in enumerate(hisgram.keys()):
                hisgram[k].append(
                    len(edges[i][edges[i] == 255]))  # xem lấy đc bao nhiêu pixel là biên , >> chèn vào hisgram
            for i, key in enumerate(hisgram.keys()):
                plt.subplot(1, 3, i + 1)
                plt.hist(hisgram[key], 30, [0, 2000])  ## lấy ra theo mảng với index = key a[i]
                plt.title(str(key))
            plt.show()


data_path = "D:/COMVISON_VIN/B4/B4/test_data/test1.jpg"
EdegeDetection(data_path).edgeshow()
