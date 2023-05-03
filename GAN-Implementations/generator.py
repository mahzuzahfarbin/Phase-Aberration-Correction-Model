class UNETGenerator(nn.Module):
    """
    This is the primary class for this AI implementation. The following is an
    overview of how this Generator network will function.

    Feature maps: Output of one filter applied to the previous layer
    Batch Norm: A method used to make training of artificial neural networks
                faster and more stable through normalization of the layers'
                inputs by re-centering and re-scaling.

        1. The input to the generator will be a 3D tensor that represents the
            anatomical 3D image spine that needs to predict phase aberration.
        2. The generator network will be a U-Net architecture, which implies
            that there must be a contracting path (i.e., encoder) and an
            expanding path (i.e., decoder) with skip connections between them.
            The purpose of the skip connections is to allow the generator to use
            information from the encoder (low resolution, but high-level
            features) and the decoder (high resolution, but low-level features)
            to produce high-level content.
        3. The contracting path will consist of several 3D convolutional layers
            with batch normalization and LeakyReLU activation functions. Each
            convolutional layer will down sample the input tensor to reduce its
            spatial dimensions while increasing the number of feature maps.
        4. The expanding path will consist of several 3D transposed convolutional
            layers with batch normalization and ReLU activation functions. Each
            transposed convolutional layer will up sample the input tensor to
            increase its spatial dimensions while decreasing the number of
            feature maps.
        5. The skip connections will connect the output of each contracting layer
            to the input of the corresponding expanding layer. The skip
            connections will concatenate the feature maps from the contracting
            path with the feature maps from the corresponding expanding path
            layer.

    TODO: Params

    TODO: Args
    """
    def __init__(self, in_channels=1, out_channels=1, num_generator_filters=64) -> \
            None:
        """
        Initialize the contracting and expanding paths
        """

        super(UNETGenerator, self).__init__()  # Calls to initializer on UNETGen

        # Contracting Path
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm3d(num_generator_filters)
        self.relu_1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_2 = nn.Conv3d(num_generator_filters, num_generator_filters * 2, kernel_size=3, stride=2, padding=1)
        self.batch_norm_2 = nn.BatchNorm3d(num_generator_filters * 2)
        self.relu_2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_3 = nn.Conv3d(num_generator_filters * 2, num_generator_filters * 4, kernel_size=3, stride=2, padding=1)
        self.batch_norm_3 = nn.BatchNorm3d(num_generator_filters * 4)
        self.relu_3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_4 = nn.Conv3d(num_generator_filters * 4, num_generator_filters * 8, kernel_size=3, stride=2, padding=1)
        self.batch_norm_4 = nn.BatchNorm3d(num_generator_filters * 8)
        self.relu_4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Expanding Path
        self.up_conv_1 = nn.ConvTranspose3d(num_generator_filters * 8, num_generator_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm_5 = nn.BatchNorm3d(num_generator_filters * 4)
        self.relu_5 = nn.ReLU(inplace=True)

        self.up_conv_2 = nn.ConvTranspose3d(num_generator_filters * 4 * 2, num_generator_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm_6 = nn.BatchNorm3d(num_generator_filters * 2)
        self.relu_6 = nn.ReLU(inplace=True)

        self.up_conv_3 = nn.ConvTranspose3d(num_generator_filters * 2 * 2, num_generator_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm_7 = nn.BatchNorm3d(num_generator_filters)
        self.relu_7 = nn.ReLU(inplace=True)

        self.up_conv_4 = nn.ConvTranspose3d(num_generator_filters, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor) -> Any:
        """
        Takes an input tensor, <input_tensor>, and applies the appropriate 
        skip connections and concatenate the feature maps.
        """
        # Contracting Path
        enc_1 = self.relu_1(self.batch_norm_1(self.conv_1(input_tensor)))
        enc_2 = self.relu_2(self.batch_norm_2(self.conv_2(enc_1)))
        enc_3 = self.relu_3(self.batch_norm_3(self.conv_3(enc_2)))
        enc_4 = self.relu_4(self.batch_norm_4(self.conv_3(enc_3)))

        # Expanding Path with Added Skip Connections
        dec_1 = self.up_conv_1(enc_4)
        dec_1 = self.relu_5(self.batch_norm_5(dec_1))
        dec_1 = torch.cat((dec_1, enc_3), dim=1)
        dec_2 = self.up_conv_2(dec_1)
        dec_2 = self.relu_6(self.batch_norm_6(dec_2))
        dec_2 = torch.cat((dec_2, enc_2), dim=1)
        dec_3 = self.up_conv_3(dec_2)
        dec_3 = self.relu_7(self.batch_norm_7(dec_3))
        dec_3 = torch.cat((dec_3, enc_1), dim=1)
        output = self.up_conv_4(dec_3)

        return output
