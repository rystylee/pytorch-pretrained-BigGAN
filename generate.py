import time
import argparse
import torchvision.utils as utils
import torchvision.transforms as transforms

import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample)


def toPIL(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model tokenizer (vocabulary)
    model = BigGAN.from_pretrained('biggan-deep-{}'.format(args.img_size)).to(device)
    model.eval()
    # print(model)

    # Prepare a input
    truncation = 0.4
    # class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=args.batch_size)
    class_vector = one_hot_from_names(['mushroom'], batch_size=args.batch_size)
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=args.batch_size)

    # # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to(device)
    class_vector = class_vector.to(device)

    # # Generate an image
    with torch.no_grad():
        start = time.time()
        output = model(noise_vector, class_vector, truncation)
        elapsed = time.time() - start
        print('elapsed: {}'.format(elapsed))

    output = output.cpu()
    output_grid = utils.make_grid(output)
    output_grid.add_(1.0).mul_(0.5)
    output_grid_img = toPIL(output_grid)
    output_grid_img.show()


if __name__ == "__main__":
    main()
