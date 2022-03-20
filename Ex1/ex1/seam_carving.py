import numpy as np


def get_greyscale_image(image, colour_wts):
    """
    Gets an image and weights of each colour and returns the image in greyscale
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (ints > 0)
    :returns: the image in greyscale
    """

    ###Your code here###
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    greyscale_image = R * colour_wts[0] + G * colour_wts[1] + B * colour_wts[2]
    ###**************###
    return greyscale_image


def reshape_bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """

    in_height, in_width, c = image.shape
    out_height, out_width = new_shape
    new_shape_with_color = (out_height, out_width, c)

    new_image = np.zeros(new_shape_with_color)

    width_ratio = in_width / out_width
    height_ratio = in_height / out_height

    def calc_floor(num):
        if num == 0:
            return 0
        else:
            return np.floor(num) if num != np.floor(num) else np.floor(num) - 1

    for i in range(out_height):
        for j in range(out_width):
            x = i * height_ratio
            y = j * width_ratio

            x_floor = int(calc_floor(x))
            y_floor = int(calc_floor(y))

            x_ceil = int(min(in_height - 1, np.ceil(x)) if x != np.ceil(x) else min(in_height - 1, np.ceil(x) + 1))
            y_ceil = int(min(in_width - 1, np.ceil(y)) if y != np.ceil(y) else min(in_width - 1, np.ceil(y) + 1))

            c1 = image[x_floor, y_floor, :]
            c2 = image[x_ceil, y_floor, :]
            c3 = image[x_floor, y_ceil, :]
            c4 = image[x_ceil, y_ceil, :]

            t = (x - x_floor) / (x_ceil - x_floor)
            s = (y - y_floor) / (y_ceil - y_floor)

            c12 = (1 - t) * c1 + t * c2
            c34 = (1 - t) * c3 + t * c4
            res = (1 - s) * c12 + s * c34

            new_image[i, j, :] = res
    new_image = new_image.astype(np.uint8)
    return new_image


def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0) 
    :returns: The gradient image
    """
    greyscale_image = get_greyscale_image(image, colour_wts)
    in_height, in_width = greyscale_image.shape
    gradient = np.zeros((in_height, in_width))

    for i in range(in_height):
        for j in range(in_width):
            if (i < (in_height - 1)) & (j < (in_width - 1)):
                first_ecq = np.square(greyscale_image[i + 1][j] - greyscale_image[i, j])
                second_ecq = np.square(greyscale_image[i, j + 1] - greyscale_image[i, j])
            else:
                first_ecq = np.square(greyscale_image[0][j] - greyscale_image[i, j])
                second_ecq = np.square(greyscale_image[i, 0] - greyscale_image[i, j])
            gradient[i, j] = np.sqrt(first_ecq + second_ecq)

    return gradient


def get_new_index_matrix(r, c):
    index_matrix = np.zeros((r, c), dtype=tuple)

    for i in range(r):
        for j in range(c):
            index_matrix[i][j] = (i, j)
    return index_matrix


def carve_column_visual(image, index_matrix):
    in_height, in_width, _ = image.shape
    M, backtrack = minimum_seam(image)
    mask = np.ones((in_height, in_width), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in reversed(range(in_height)):
        mask[i, j] = False
        j = backtrack[i, j]

    index_matrix = index_matrix[mask].reshape((in_height, in_width - 1))

    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = image[mask].reshape((in_height, in_width - 1, 3))
    return index_matrix, img


def calculate_visualise_seams(index_matrix, new_img, number_of_vertical):
    for i in range(number_of_vertical):
        index_matrix, new_img = carve_column_visual(new_img, index_matrix)

    return index_matrix, new_img


def get_valid_mask_matrix(mask, index_matrix):
    pass


def paint_the_seam(image, index_matrix, colour):
    in_height, in_width, _ = image.shape
    index_height, index_width = index_matrix.shape
    demo_img = image.copy()
    mask = np.ones((in_height, in_width), dtype=np.bool)

    for i in range(index_height):
        for j in range(index_width):
            (x, y) = index_matrix[i][j]
            mask[x][y] = False

    for i in range(in_height):
        for j in range(in_width):
            if mask[i][j]:
                demo_img[i][j] = colour

    return demo_img


def visualise_seams(image, new_shape, show_horizontal, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param show_horizontal: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    if show_horizontal:
        image = np.rot90(image)
    in_height, in_width, _ = image.shape
    new_img = image.copy()

    index_matrix = get_new_index_matrix(in_height, in_width)
    vertical_seams_to_remove = in_height - new_shape[0]
    index_matrix, new_img = calculate_visualise_seams(index_matrix, new_img, vertical_seams_to_remove)

    image = paint_the_seam(image, index_matrix, colour)
    if show_horizontal:
        image = np.rot90(image, 3)

    return image


def minimum_seam(image):
    in_height, in_width, _ = image.shape
    gradient_map = gradient_magnitude(image, [0.299, 0.587, 0.114])

    M = gradient_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, in_height):
        for j in range(0, in_width):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_value = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_value = M[i - 1, idx + j - 1]
            M[i, j] += min_value

    return M, backtrack


def reshape_seam_crarving(image, new_shape, carving_scheme):
    """
    Resizes an image to new shape using seam carving
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    new_img = image.copy()

    for i in range(in_width - new_shape[0]):
        new_img = carve_column(new_img)
    new_img = np.rot90(new_img)
    for i in range(in_height - new_shape[1]):
        new_img = carve_column(new_img)

    seam_image = np.rot90(new_img, 3)
    return seam_image


def carve_column(image):
    in_height, in_width, _ = image.shape

    M, backtrack = minimum_seam(image)
    mask = np.ones((in_height, in_width), dtype=np.bool)
    j = np.argmin(M[-1])

    for i in reversed(range(in_height)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = image[mask].reshape((in_height, in_width - 1, 3))

    return img
