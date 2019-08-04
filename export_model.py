import argparse

import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model tokenizer (vocabulary)
    model = BigGAN.from_pretrained('biggan-deep-{}'.format(args.img_size)).to(device)
    model.eval()
    print(model)

    # Prepare a input
    noise_vector = torch.ones(1, args.z_dim).to(device)
    class_vector = torch.ones(1, args.n_classes).to(device)
    truncation = torch.ones(1).to(device)

    traced_script_module = torch.jit.trace(model, (noise_vector, class_vector, truncation))
    output = traced_script_module(noise_vector, class_vector, truncation)

    name = 'BigGAN_{}.pt'.format(args.img_size)
    traced_script_module.save(name)
    print('Succeed to save traced script module!')


if __name__ == "__main__":
    main()
