import math

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


# img=original_img
def reshape_bilinear(image, new_shape):
    # """
    # Resizes an image to new shape using bilinear interpolation method
    # :param image: The original image
    # :param new_shape: a (height, width) tuple which is the new shape
    # :returns: the image resized to new_shape
    # """
    # in_height, in_width, c = image.shape
    # out_height, out_width = new_shape
    # new_shape_with_color = (out_height, out_width, c)
    #
    # new_image = np.zeros(new_shape_with_color)
    #
    # width_ratio = (in_width) / out_width
    # height_ratio = (in_height) / out_height
    #
    # def calc_flor(num):
    #     if num == 0:
    #         return 0
    #     else:
    #         return np.floor(num) if num != np.floor(num) else np.floor(num) - 1
    #
    # def calc_ciel(num,h_or_w):
    #
    #     return min(h_or_w - 1, np.ceil(num)) if x != np.ceil(num) else min(h_or_w - 1, np.ceil(num) + 1)
    #
    # for i in range(out_height):
    #     for j in range(out_width):
    #         # map the coordinates back to the original image
    #         x = i * height_ratio
    #         y = j * width_ratio
    #         # calculate the coordinate values for 4 surrounding pixels.
    #
    #         x_floor = int(calc_flor(x))
    #         x_ceil = int(calc_ciel(x,in_height))
    #         y_floor = int(calc_flor(y))
    #         y_ceil = int(calc_ciel(y,in_width))
    #
    #         c1 = image[x_floor, y_floor, :]
    #         c2 = image[x_ceil, y_floor, :]
    #         c3 = image[x_floor, y_ceil, :]
    #         c4 = image[x_ceil, y_ceil, :]
    #
    #         t = (x - x_floor) / (x_ceil - x_floor)
    #         s = (y - y_floor) / (y_ceil - y_floor)
    #         c12 = (1 - t) * c1 + t * c2
    #         c34 = (1 - t) * c3 + t * c4
    #         res = (1 - s) * c12 + s * c34
    #         new_image[i, j, :] = res
    #
    # ###**************####
    # return new_image

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

    return new_image.astype(np.uint8)


def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0) 
    :returns: The gradient image
    """

    ###Your code here###

    greyscale_image = get_greyscale_image(image, colour_wts)
    height, width = greyscale_image.shape
    gradient = np.zeros((height, width))
    for i in range(height - 1):
        for j in range(width - 1):
            first_ecq = np.square(greyscale_image[i + 1][j] - greyscale_image[i, j])
            second_ecq = np.square(greyscale_image[i, j + 1] - greyscale_image[i, j])
            gradient[i, j] = np.sqrt(first_ecq + second_ecq)
    ###**************###
    return gradient


def paint_seams(new_img, image, colour):
    in_height, in_width, _ = image.shape
    for i in range(in_width):
        for j in range(in_width):
            if new_img[i, j] == [256, 256, 256]:
                image[i, j] = colour
    return image


def visualise_seams(image, new_shape, carving_scheme, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    ###Your code here###
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_img = image.copy()
    number_of_seams_to_remove = in_width - out_width
    for i in range(number_of_seams_to_remove):
        new_img = carve_column_visual(new_img)
    seam_image = paint_seams(new_img, image, colour)
    # seam_image = new_img
    return seam_image

    return seam_image


def minimum_seam(image):
    r, c, _ = image.shape
    energy_map = gradient_magnitude(image, [0.299, 0.587, 0.114])

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    return M, backtrack


# def carve_column(img, mask):
#     r, c, _ = img.shape
#
#     M, backtrack = minimum_seam(img)
#     ##mask = np.ones((r, c), dtype=np.bool)
#     j = np.argmin(M[-1])
#
#     for i in reversed(range(r)):
#         mask[i, j] = False
#         j = backtrack[i, j]
#
#     ##mask = np.stack([mask] * 3, axis=2)
#
#     # Delete all the pixels marked False in the mask,
#     # and resize it to the new image dimensions
#     ##img = img[mask].reshape((r, c - 1, 3))
#
#     return img, mask
#     ###**************###
#     # return seam_image
#     pass

def reshape_seam_crarving(image, new_shape, carving_scheme):
    """
    Resizes an image to new shape using seam carving
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :returns: the image resized to new_shape
    """
    r, c, _ = image.shape
    new_img = image.copy()

    for i in range(c - new_shape[0]):
        new_img = carve_column(new_img)

    seam_image = new_img
    return seam_image


def minimum_seam(image):
    r, c, _ = image.shape
    energy_map = gradient_magnitude(image, [0.299, 0.587, 0.114])

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    return M, backtrack


def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    new_img = img[mask]
    img = img[mask].reshape((r, c - 1, 3))

    return img


def carve_column_visual(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        img[i, j] = np.array([256, 256, 256])
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    # img = img[mask].reshape((r, c - 1, 3))

    return img
