from collections import defaultdict

"""
Structural similarity index implementation
https://en.wikipedia.org/wiki/Structural_similarity
"""


def _get_variance(pixel_count, average):
    variance = 0.0
    for pixel in pixel_count:
        residual = pixel - average
        variance += pixel_count[pixel] * residual ** 2
    return variance


def get_ssim_components(img1, img2, tile_size, pixel_len, width, height, c_1, c_2):
    '''
    Iteratively calculates SSIM for tiles of size tile_size
    '''
    ssim_sum = 0
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            box = (x, y, x + 7, y + 7)
            tile_0, tile_1 = img1.crop(box), img2.crop(box)

            pixel0, pixel1 = tile_0.getdata(band=0), tile_1.getdata(band=0)
            color_count_0, color_count_1 = defaultdict(int), defaultdict(int)
            covariance = 0.0
            for i1, i2 in zip(pixel0, pixel1):
                color_count_0[i1] += 1
                color_count_1[i2] += 1
                covariance += i1 * i2

            mean0 = sum(pixel0) / pixel_len
            mean1 = sum(pixel1) / pixel_len

            covariance = (covariance - sum(pixel0) * sum(pixel1) / pixel_len) / pixel_len

            var0 = _get_variance(color_count_0, mean0)/pixel_len
            var1 = _get_variance(color_count_1, mean1)/pixel_len

            ssim_sum += (2.0 * mean0 * mean1 + c_1) * (2.0 * covariance + c_2) /\
                        (mean0**2 + mean1**2 + c_1) / (var0 + var1 + c_2)

    return ssim_sum


def compare_ssim(img1, img2, sub_size=7):
    L = 2**8 - 1
    c_1 = (L * 0.01) ** 2
    c_2 = (L * 0.03) ** 2
    pixel_len = sub_size * sub_size
    w, h = img1.size
    w = w // sub_size * sub_size
    h = h // sub_size * sub_size
    return get_ssim_components(img1, img2, sub_size, pixel_len, w, h, c_1, c_2) * pixel_len / (
            len(img1.mode) * w * h)
