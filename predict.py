import argparse
import json
from my.data_set import Transform
from my.network import Network


def main():
    # Parse commandline options and arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', action='store', type=str)
    parser.add_argument('model_file', action='store', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--arch', action='store', default='vgg13', type=str)
    parser.add_argument('--hidden-units', action='store', default=2048, type=int)
    parser.add_argument('--top_k', action='store', default=5, type=int)
    parser.add_argument('--category_names', action='store', default='cat_to_name.json', type=str)
    args = parser.parse_args()

    # Define data_set and network.
    transform = Transform()
    network = Network(arch=args.arch, hidden_units=args.hidden_units)

    # Load network.
    network.load(args.model_file)

    # Set device to gpu when it is available.
    if args.gpu:
        network.device = 'cuda'

    # Predict.
    probability, classes = network.predict(args.image_file, transform, topk=args.top_k)

    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[clss] for clss in classes]

    for i, (prob, name) in enumerate(zip(probability, names)):
        print("No: {}".format(i+1))
        print("Probability: {:.5f}".format(prob))
        print("name: {}".format(name))
        print()


if __name__ == '__main__':
    main()
