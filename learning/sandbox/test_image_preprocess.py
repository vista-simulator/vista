from PIL import Image, ImageEnhance


kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}


def adjust_temp(image, temp):
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', matrix)


def adjust_gamma(image, gamma=1.0):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(gamma)


def adjust_color(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def rgba2rgb(image):
    new_image = Image.new('RGB', image.size, (255, 255, 255))
    new_image.paste(image, mask=image.split()[3])
    return new_image

image = Image.open('center.png')
image_pp = image
image_pp = adjust_temp(rgba2rgb(image), 5000)
image_pp = adjust_gamma(image_pp, 1.5)
image_pp = adjust_color(image_pp, 1.5)
image_pp.save('test.png')
