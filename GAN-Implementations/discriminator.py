class WassersteinDiscriminator(nn.Module):
    """
    This class serves as the discriminator network in the architecture. The
    following is an overview of how this Discriminator network will function.

    Unlike a traditional GAN, we will be setting up a WGAN, whereby a
    Wasserstein loss is designed in contrast to setting up binary cross-entropy
    loss. The discriminator is designed to take in a 2D image that is curated
    by the functioning generator network. The input image is then passed through
    a series of convolutional layers, with each layer followed by a batch
    normalization layer and a LeakyReLU activation function.
    The final layer outputs a single scalar value, representing the
    possibility that the input image is real
    """

    def __init__(self, in_channels=1, num_discrim_filters=64) -> None:
        """
        Initialize the discriminator network architecture.
        """
        super(WassersteinDiscriminator, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, num_discrim_filters, kernel_size=4, stride=2, padding=1)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_2 = nn.Conv2d(num_discrim_filters, num_discrim_filters * 2, kernel_size=4, stride=2, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_discrim_filters * 2)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_3 = nn.Conv2d(num_discrim_filters * 2, num_discrim_filters * 4, kernel_size=4, stride=2, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(num_discrim_filters * 4)
        self.relu_3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_4 = nn.Conv2d(num_discrim_filters * 4, num_discrim_filters * 8, kernel_size=4, stride=2, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(num_discrim_filters * 8)
        self.relu_4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_5 = nn.Conv2d(num_discrim_filters * 8, out_channels=1, kernel_size=4, stride=1, padding=0)

    def forward(self, input_tensor) -> Any:
        """
        Takes an input tensor <input_tensor> and passes it through a series of
        convolutional layers, batch normalization layers, and leaky ReLU
        activation functions to learn features from the input image. These
        features are then flattened and passed through a fully connected layer
        to obtain a scalar output.
        """
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.relu1(input_tensor)
        input_tensor = self.conv2(input_tensor)
        input_tensor = self.bn2(input_tensor)
        input_tensor = self.relu2(input_tensor)
        input_tensor = self.conv3(input_tensor)
        input_tensor = self.bn3(input_tensor)
        input_tensor = self.relu3(input_tensor)
        input_tensor = self.conv4(input_tensor)
        input_tensor = self.bn4(input_tensor)
        input_tensor = self.relu4(input_tensor)
        input_tensor = self.conv5(input_tensor)

        return input_tensor.view(-1, 1)
