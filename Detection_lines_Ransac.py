import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread('D:/COMVISON_VIN/B4/B4/test_data/test1.jpg')

# Phát hiện cạnh và biên trong ảnh
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Khởi tạo các tham số cho RANSAC
n_iterations = 1000
threshold_distance = 10
inlier_ratio = 0

# Tính số điểm cần thiết để tính toán mô hình đường thẳng
min_points_for_line = 2

# Khởi tạo các biến cho mô hình đường thẳng tốt nhất
best_line = None
best_line_inliers = []

# Thực hiện RANSAC để tìm đường thẳng tốt nhất
for i in range(n_iterations):
    # Lấy mẫu ngẫu nhiên từ các điểm biên
    sample_points = np.random.randint(0, len(edges), min_points_for_line)
    sample = np.array([(x, y) for (y, x) in np.argwhere(edges)])
    sample = sample[sample_points]

    # Tính toán mô hình đường thẳng từ các điểm lấy mẫu
    line = cv2.fitLine(sample, cv2.DIST_L2, 0, 0.01, 0.01)

    # Tính toán khoảng cách từ các điểm biên đến đường thẳng
    distances = np.abs((sample - line[:2]).dot(line[2:]).flatten())

    # Tính toán số điểm trong tập dữ liệu ban đầu có thể được giải thích bằng mô hình
    inliers = sample[distances < threshold_distance]
    inlier_ratio_current = len(inliers) / len(sample)

    # Kiểm tra xem mô hình tìm được có tốt hơn mô hình tốt nhất hiện tại không
    if inlier_ratio_current > inlier_ratio:
        inlier_ratio = inlier_ratio_current
        best_line = line
        best_line_inliers = inliers

# Vẽ đường thẳng tốt nhất trên ảnh gốc
if len(best_line_inliers) == 0:
    print("No inliers found!")
    # handle the empty inliers list here
else:
    x0, y0 = tuple(map(int, np.mean(best_line_inliers, axis=0)))
dx, dy = best_line[2:]
cv2.line(img, (int(x0 - dx * 1000), int(y0 - dy * 1000)), (int(x0 + dx * 1000), int(y0 + dy * 1000)), (0, 0, 255), 2)

# Hiển thị ảnh
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
