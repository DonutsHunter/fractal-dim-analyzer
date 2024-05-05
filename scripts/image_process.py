import numpy as np
import tempfile
import cv2


"""
# OG contour detector
def image_separation(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(image_file.read())

    image = text_remove(temp_filename)

    # Удаляем временный файл
    temp_file.close()

    # Обработка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 5, 75, 75)
    
    block_size = 629
    constant = 3
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        constant
    )

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров
    filtered_contours = []
    image_height, image_width = threshold.shape
    min_contour_area = 1000 

    for contour in contours:
        # Вычисление площади контура
        contour_area = cv2.contourArea(contour)

        # Проверка площади контура
        if contour_area < min_contour_area:
            continue

        # Проверка координат контура
        is_on_edge = False
        for point in contour:
            x, y = point[0]
            if x == 0 or x == image_width - 1 or y == 0 or y == image_height - 1:
                is_on_edge = True
                break

        if not is_on_edge:
            filtered_contours.append(contour)

    areas = [cv2.contourArea(contour) for contour in filtered_contours]
    largest_index = np.argmax(areas)
    largest_contour = [filtered_contours[largest_index]]

    # Создаем новое изображение
    processed_image = np.ones_like(image, dtype=np.uint8) * 255  # Создаем изображение с белым фоном
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2BGRA)  # Преобразуем в BGRA
    processed_image[:, :, 3] = 255  # Устанавливаем полную непрозрачность

    # Рисование контуров на обработанном изображении
    cv2.drawContours(processed_image, largest_contour, -1, (0, 0, 0, 255), thickness=2)
    
    return processed_image
"""


"""
def image_separation(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(image_file.read())

    image = cv2.imread(temp_filename)

    # Удаляем временный файл
    temp_file.close()

    # Обработка изображения
    filterd_image = cv2.medianBlur(image, 7)
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = 100

    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

    # Поиск контуров на изображении
    contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем новое изображение
    processed_image = np.ones_like(image, dtype = np.uint8) * 255  # Создаем изображение с белым фоном
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2BGRA)  # Преобразуем в BGRA
    processed_image[:, :, 3] = 255  # Устанавливаем полную непрозрачность

    # Рисование контуров на обработанном изображении
    cv2.drawContours(processed_image, contours, -1, (0, 0, 0, 255), 1)

    return processed_image
"""


"""
def image_separation(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(image_file.read())

    image = cv2.imread(temp_filename)

    # Удаляем временный файл
    temp_file.close()

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация изображения (черно-белое)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Нахождение контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создание белого фона для цветного изображения
    processed_image = np.zeros_like(image)
    
    # Нарисовать контуры на белом фоне
    cv2.drawContours(processed_image, contours, -1, (255, 255, 255), 2)
    
    # Объединение оригинального изображения и изображения с контурами
    result_image = cv2.addWeighted(image, 0.7, processed_image, 0.3, 0)
    
    return result_image
"""


def image_separation(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(image_file.read())

    image = cv2.imread(temp_filename)

    # Удаляем временный файл
    temp_file.close()

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Нахождение контуров
    contours, _ = cv2.findContours(image=thresh, 
                                        mode=cv2.RETR_EXTERNAL, 
                                        method=cv2.CHAIN_APPROX_SIMPLE
                                        )
    
    image_copy = image.copy()
    
    # Нарисовать внешние и внутренние контуры на белом фоне
    cv2.drawContours(image=image_copy, 
                    contours=contours, 
                    contourIdx=-1, 
                    color=(0, 0, 0), 
                    thickness=1, 
                    lineType=cv2.LINE_AA
                    )

    ########################################################

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10., tileGridSize=(8,8))

    lab = cv2.cvtColor(image_copy, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels

    ########################################################

    dst = cv2.normalize(lab[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blurred = cv2.GaussianBlur(dst, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 150)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)

    # Нахождение контуров
    contours, _ = cv2.findContours(image=dilate, 
                                        mode=cv2.RETR_EXTERNAL, 
                                        method=cv2.CHAIN_APPROX_SIMPLE
                                        )
    
    image_copy_copy = image_copy.copy()
    
    # Нарисовать внешние и внутренние контуры на белом фоне
    cv2.drawContours(image=image_copy_copy, 
                    contours=contours, 
                    contourIdx=-1, 
                    color=(0, 0, 0), 
                    thickness=1, 
                    lineType=cv2.LINE_AA
                    )
    
    ########################################################

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image_copy_copy, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Нахождение контуров
    contours, _ = cv2.findContours(image=thresh, 
                                        mode=cv2.RETR_EXTERNAL, 
                                        method=cv2.CHAIN_APPROX_SIMPLE
                                        )
    
    processed_image = np.zeros_like(image)
    processed_image.fill(255)
    
    # Нарисовать внешние и внутренние контуры на белом фоне
    cv2.drawContours(image=processed_image, 
                    contours=contours, 
                    contourIdx=-1, 
                    color=(0, 0, 0), 
                    thickness=1, 
                    lineType=cv2.LINE_AA
                    )

    # Вывод результата
    return processed_image