from typing import *

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


def get_image(return_png: bool = False, driver: str = "ndarray"):
    assert driver in ["pil", "ndarray"]

    def process_show(show_func: Callable) -> Callable:
        def show_with_png(*args, **kwargs):
            show_func(*args, **kwargs)
            import matplotlib.backends.backend_agg as bagg
            canvas = bagg.FigureCanvasAgg(plt.gcf())
            canvas.draw()
            png, (width, height) = canvas.print_to_buffer()
            png = np.frombuffer(png, dtype=np.uint8).reshape((height, width, 4))

            if driver == 'pil':
                return Image.fromarray(png)
            else:
                return png

        if return_png:
            return show_with_png
        else:
            return show_func

    return process_show


ImageType = TypeVar("ImageType", np.ndarray, List[np.ndarray], None)
TitleType = TypeVar("TitleType", str, List[str], None)


# @get_image(return_png=True, driver="pil")
@get_image(return_png=True, driver="ndarray")
def show(image: ImageType, title: TitleType = None) -> None:
    # make images
    assert image is not None, f"image cannot be None"
    if isinstance(image, np.ndarray):
        assert image.ndim in [3, 4], f"Wrong Shape, should be [channel, width, height]" \
                                     f" or [image, channel, width, height], but you offered {image.shape}"
        assert image.shape[-3] in [1, 3], f"Wrong channel, should be [channel, width, height]" \
                                          f" or [image, channel, width, height], but you offered {image.shape}"
        if image.ndim == 3:
            image = [image]
        elif image.ndim == 4:
            image = [image[i, ...] for i in range(len(image))]
    else:
        for i in image:
            assert i.ndim == 3, f"Wrong shape, should be List[np.ndarray[channel, width, height]]," \
                                f" but you offered {i.shape}"
            assert i.shape[0] == 3, f"Wrong shape, should be List[np.ndarray[channel, width, height]]" \
                                    f" but you offered {i.shape}"

    # make title
    assert isinstance(title, (str, list)) or title is None, f"Wrong type, should be str or List[str]"
    if isinstance(title, str) or title is None:
        if len(image) > 1:
            repeats = len(image)
        else:
            repeats = 1
        title = [title if isinstance(title, str) else ""] * repeats
    elif isinstance(title, list):
        for i in title:
            assert isinstance(i, str), f"Wrong type, should be List[str]"

    import math
    grid_r = int(math.sqrt(len(image)))
    grid_c = math.ceil(len(image) / grid_r)

    fig, ax = plt.subplots(nrows=grid_r, ncols=grid_c, figsize=(3 * grid_r, 4 * grid_c), layout="tight")
    if len(image) == 1:
        ax = [[ax]]
    elif grid_r == 1:
        ax = [ax]

    from matplotlib.axes import Axes
    ax: Union[List[Axes], List[List[Axes]]]
    for idx in range(len(image)):
        row = idx // grid_r
        col = idx - row * grid_r
        ax[row][col].imshow(image[idx].transpose(1, 2, 0))
        ax[row][col].set_title(title[idx])


if __name__ == "__main__":
    single_image = np.random.random(size=(3, 256, 256))
    multiple_image = np.random.random(size=(16, 3, 256, 256))
    image_list = [np.random.random(size=(3, 256, 256)) for i in range(16)]

    title_none = None

    # show(image=single_image)
    # show(image=image_list, title=[f"random_{i}" for i in range(len(image_list))]).show()
    img = show(image=multiple_image)
    print(img.shape)
    plt.show()
