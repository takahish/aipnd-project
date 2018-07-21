import torch
from torchvision import models
from collections import OrderedDict
from PIL import Image


class Network(object):
    """Network of transfer learning.
    """
    def __init__(self, hidden_units=2048, device='cpu'):
        """Initialize Network

        Args:
            hidden_units: int (>0). classifier hidden units.
            device: string ('cpu' or 'gpu').
        """
        self.__model = models.vgg19_bn(pretrained=True)

        # Freeze parameters
        for parameter in self.__model.parameters():
            parameter.required_grad = False

        # Define classifier
        classifier = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(25088, hidden_units)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('drop1', torch.nn.Dropout(0.5)),
            ('fc2', torch.nn.Linear(hidden_units, 102)),
            ('output', torch.nn.LogSoftmax(dim=1))
        ]))

        # Replace classifier
        self.__model.classifier = classifier

        # Set device
        self.__device = device
        self.__model.to(self.__device)

        # Set criterion
        self.__criterion = torch.nn.NLLLoss()

        # Set optimizer
        self.__optimizer = None

    @property
    def model(self):
        """Getter for model.
        """
        return self.__model

    @property
    def device(self):
        """Getter for device.
        """
        return self.__device

    @device.setter
    def device(self, device):
        """Setter for device.
        """
        self.__device = device
        self.__model.to(self.__device)

    @property
    def criterion(self):
        """Getter for criterion.
        """
        return self.__criterion

    @property
    def optimizer(self):
        """Getter for optimizer.
        """
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, learning_rate):
        """Setter for optimizer.
        """
        self.__optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

    def train(self, data_set, learning_rate=0.0003, epochs=3, print_every=40):
        """Training Network.

        Args:
            learning_rate: float.
            epochs: int (>0).
            print_every: int (>0).
        """
        # Set optimizer
        self.optimizer = learning_rate

        steps = 0
        for e in range(epochs):
            # Make sure network is in train mode for training
            self.model.train()
            running_loss = 0

            for inputs, labels in data_set.train_dataloader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Forward and backward pass
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval model for inference
                    self.model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.__validation(data_set.valid_dataloader)

                    print(
                        "Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(test_loss),
                        "Validation accuracy: {:.3f}".format(accuracy)
                    )

                    running_loss = 0

                    # Make sure training is back on
                    self.model.train()

        # Set class to index.
        self.model.class_to_idx = data_set.train_data_set.class_to_idx

        # Set index to class.
        self.model.idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}

    def __validation(self, data_loader):
        """Evaluating accuracy for validation or test data set.

        Args:
            data_loader: torch.utils.data.DataLoader.

        Returns:
            test_loss: float. Loss of validation or test data set.
            accuracy: float. Accuracy of validation or test data set.
        """
        test_loss = 0
        accuracy = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model.forward(inputs)
            test_loss += self.criterion(outputs, labels).item()

            probability = torch.exp(outputs)
            equality = (labels.data == probability.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss / len(data_loader), accuracy / len(data_loader)

    def save(self, file_path, device='cpu'):
        """Saving model state into file path.

        Args:
            file_path: string.
        """
        self.device = device

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'idx_to_class': self.model.idx_to_class
        }

        torch.save(checkpoint, file_path)

    def load(self, file_path, device='cpu'):
        """Loading model state from file path.

        Args:
             file_path: string.
        """
        self.device = device

        checkpoint = torch.load(file_path)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model.idx_to_class = checkpoint['idx_to_class']

    def predict(self, image_path, transform, topk=5):
        """Predicting image.

        Args:
            image_path: string. Image file path.
            trainsform: my.data_set.Transform.
            topk: int.

        Returns:
            probability of top_k class.
            top_k class.
        """
        # Process image
        image = Image.open(image_path)
        image = transform.test_transform(image).numpy()
        image = torch.unsqueeze(torch.from_numpy(image), 0)
        image = image.to(self.device)

        # Predict
        self.model.eval()
        outputs = self.model.forward(image)

        # Get top-k
        log_liklihood, labels = outputs.topk(topk)
        probability = torch.exp(log_liklihood)

        return probability.tolist()[0], [str(self.model.idx_to_class[label]) for label in labels.tolist()[0]]
