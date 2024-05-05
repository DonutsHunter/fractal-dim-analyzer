from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import tempfile
import shutil
import cv2


def preprocess(input_matrix):
    sum_matrix = np.zeros_like(input_matrix, dtype=np.uint32)
    m, n = input_matrix.shape

    for i in range(m):
        for j in range(n):
            sum_matrix[i, j] = input_matrix[i, j]

            if i > 0:
                sum_matrix[i, j] += sum_matrix[i - 1, j]

            if j > 0:
                sum_matrix[i, j] += sum_matrix[i, j - 1]

            if i > 0 and j > 0:
                sum_matrix[i, j] -= sum_matrix[i - 1, j - 1]

    return sum_matrix


def box_count(sum_matrix, box_width):
    m, n = sum_matrix.shape
    count = 0

    for i in range(0, m, box_width):
        for j in range(0, n, box_width):
            box_sum = calc_sum(
                sum_matrix,
                i,
                j,
                min(i + box_width - 1, m - 1),
                min(j + box_width - 1, n - 1)
            )
            if box_sum > 0:
                count += 1

    return count


def calc_sum(sum_matrix, i, j, k, m):
    box_sum = sum_matrix[k, m]

    if i > 0:
        box_sum -= sum_matrix[i - 1, m]

    if j > 0:
        box_sum -= sum_matrix[k, j - 1]

    if i > 0 and j > 0:
        box_sum += sum_matrix[i - 1, j - 1]

    return box_sum


def fractal_dim(image_file, box_width_start, box_width_end, box_width_increment):
    # Read imagefile as grayscale image
    image = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

    # Convert to binary image
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Get presum for the input matrix
    sum_matrix = preprocess(image)

    # Amount iterations of box counting
    num_iter = int((box_width_end - box_width_start) / box_width_increment)
    y = np.zeros(num_iter)
    x = np.zeros(num_iter)
    num = 0

    for box_width in np.arange(box_width_start, box_width_end, box_width_increment):
        count = box_count(sum_matrix, box_width)
        num += 1

        y[num - 1] = np.log(count)
        x[num - 1] = np.log(1 / box_width)

    fit = np.polyfit(x, y, 1)
    fractal_dimension = fit[0]

    return fractal_dimension


def fractal_dim_graphics(image_file, box_width_start, box_width_end, box_width_increment):
    # Read imagefile as grayscale image
    image = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

    initial_image = image.copy()

    # Convert to binary image
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Get presum for the input matrix
    sum_matrix = preprocess(image)

    # Amount iterations of box counting
    num_iter = int((box_width_end - box_width_start) / box_width_increment)
    y = np.zeros(num_iter)
    x = np.zeros(num_iter)
    num = 0

    fig, ax = plt.subplots()

    temp_dir = tempfile.mkdtemp()
    images = []

    box_counting = {}

    for box_width in np.arange(box_width_start, box_width_end, box_width_increment):
        count = box_count(sum_matrix, box_width)
        num += 1

        y[num - 1] = np.log(count)
        x[num - 1] = np.log(1 / box_width)

        res = [f'№{num}: Размер ячейки = {box_width}, Кол-во ячеек = {count}']
        box_counting[num] = res

        ax.imshow(initial_image, cmap='gray')

        for i in range(0, initial_image.shape[0], box_width):
            for j in range(0, initial_image.shape[1], box_width):
                if calc_sum(sum_matrix, i, j, min(i + box_width - 1, initial_image.shape[0] - 1),
                            min(j + box_width - 1, initial_image.shape[1] - 1)) > 0:
                    rect = plt.Rectangle(
                        xy=(j, i),
                        width=box_width,
                        height=box_width,
                        linewidth=0.5,
                        edgecolor='r',
                        facecolor='pink',
                        alpha=0.5
                    )
                    ax.add_patch(rect)

        ax.axis('off')
        plt.savefig(f'{temp_dir}/image{num:03d}.png')
        ax.cla()

        img = Image.open(f'{temp_dir}/image{num:03d}.png')
        images.append(img)

    plt.close()

    output_path = './assets/fd_grid.gif'

    images[0].save(output_path, save_all=True, append_images=images[1:], duration=200, loop=0)

    shutil.rmtree(temp_dir)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x, y, "ko", markerfacecolor="pink")
    ax.set_xlabel("log (1/length)")
    ax.set_ylabel("log (number of boxes)")

    # Fractal dimension is the slope
    fit = np.polyfit(x, y, 1)
    fractal_dimension = fit[0]

    ax.set_title(f"Fractal Dimension by Box Counting\n D = {fractal_dimension}")
    ax.plot(x, np.polyval(fit, x), color="b", linewidth=1)

    return fig, box_counting, fractal_dimension