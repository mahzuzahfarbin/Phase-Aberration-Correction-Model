
class Discriminator(nn.Module):
    """Perform image discrimination (i.e., output a scalar value that is used
    to determine if image is real or fake."""

    def __init__(self, in_channels=1, out_channels=1, num_filters=64):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_filters * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(num_filters * 4)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(num_filters * 8)
        self.leakyrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(num_filters * 8, out_channels, kernel_size=4, stride=1, padding=0)

    def forward(self, input_tensor):
        """Computation graph of the discriminator network (i.e., flattens 2D image
        to 1D tensor)."""
        x = self.conv1(input_tensor)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)

        return x.view(-1, 1)
