import random
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

try:
    from nltk.corpus import words
    word_list = words.words()
except:
    import nltk
    nltk.download('words')


def showImagesHorizontally(images, figsize=(16, 16)):
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        a = fig.add_subplot(1, len(images), i + 1)
        image = images[i]
        plt.imshow(image)
        plt.axis('off')


def random_place(height=256, width=256):
    y = random.randint(0, height)
    x = random.randint(0, width - 30)
    return x, y


def random_text(min_n_lines=1, max_n_lines=4):
    return '\n'.join(random.choices(word_list, k=random.randint(min_n_lines, max_n_lines)))


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def random_font(min_font_size=8, max_font_size=32, font='fonts/Monaco.ttf'):
     return ImageFont.truetype(font, size=random.randint(min_font_size, max_font_size))



def random_draw_text(draw, min_n_texts=1, max_n_texts=4,
                     height=256, width=256,
                     min_n_lines=1, max_n_lines=4,
                     min_font_size=8, max_font_size=32):
    for _ in range(random.randint(min_n_texts, max_n_texts)):
        xy = random_place(height, width)
        text = random_text(min_n_lines, max_n_lines)
        color = random_color()
        font = random_font(min_font_size, max_font_size)
        draw.text(xy, text, fill=color, font=font)


def generate_random_image(min_n_texts=1, max_n_texts=4,
                          height=256, width=256,
                          min_n_lines=1, max_n_lines=4,
                          min_font_size=8, max_font_size=32):
    background_color = random_color()
    image = Image.new('RGB', (height, width), color=background_color)
    draw = ImageDraw.Draw(image)
    random_draw_text(
        draw,
        min_n_texts=min_n_texts,
        max_n_texts=max_n_texts,
        height=height,
        width=width,
        min_n_lines=min_n_lines,
        max_n_lines=max_n_lines,
        min_font_size=min_font_size,
        max_font_size=max_font_size
    )
    return image
