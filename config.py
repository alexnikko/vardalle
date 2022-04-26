from munch import Munch


generate_params = dict(
    min_n_texts=1,
    max_n_texts=1,
    height=256,
    width=256,
    min_n_lines=1,
    max_n_lines=1,
    min_font_size=26,#16,
    max_font_size=26#32
)
generate_params = Munch.fromDict(generate_params)