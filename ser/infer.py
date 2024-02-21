from ser.data import test_dataloader
from ser.transforms import transforms, normalize
import torch

def do_infer(model, image):
    model.eval()
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    pixels = image[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}, with confidence {confidence * 100:.2f}%")
    # 



def select_image(label):
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)

def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
