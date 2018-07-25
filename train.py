import argparse
from my.data_set import Transform, DataSet
from my.network import Network


def main():
    # Parse commandline options and arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action='store', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--arch', action='store', default='vgg13', type=str)
    parser.add_argument('--hidden_units', action='store', default=2048, type=int)
    parser.add_argument('--learning_rate', action='store', default=0.0003, type=float)
    parser.add_argument('--epochs', action='store', default=3, type=int)
    parser.add_argument('--print_every', action='store', default=40, type=int)
    parser.add_argument('--save_file', action='store', default='checkpoint.pth', type=str)
    args = parser.parse_args()

    # Define data_set and network.
    transform = Transform()
    data_set = DataSet(args.data_directory, transform)
    network = Network(arch=args.arch, hidden_units=args.hidden_units)

    # Set device to gpu when it is available.
    if args.gpu:
        network.device = 'cuda'

    # Train network.
    network.train(data_set, learning_rate=args.learning_rate, epochs=args.epochs, print_every=args.print_every)

    # Save network
    network.save(args.save_file)


if __name__ == '__main__':
    main()
