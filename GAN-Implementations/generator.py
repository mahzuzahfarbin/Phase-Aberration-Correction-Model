class Generator(nn.Module):
    """The generator network which maintains an encoder-decoder architecture
    for down-sampling a 3D volume to a lower-dimensional representation.
    """
    def __init__(self, in_channels=1, out_channels=1, num_filters=64):
        """Initialize the generator"""
        super(Generator, self).__init__()
        # Encoder
        self.conv1 = nn.Conv3d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(num_filters * 8, num_filters * 16, kernel_size=4, stride=2, padding=1)
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(num_filters * 16, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(num_filters, out_channels, kernel_size=4, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Computation graph of the generator network (i.e., how the input tensor
        flows through the layers in initializer)"""
        x1 = self.conv1(input_tensor)
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        # Decoder
        y1 = self.relu(self.deconv1(x5))
        y2 = self.relu(self.deconv2(y1))
        y3 = self.relu(self.deconv3(y2))
        y4 = self.relu(self.deconv4(y3))
        output = self.tanh(self.deconv5(y4))

        return output
