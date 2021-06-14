from torch import nn

class FCNet(nn.Module):
    def __init__(
        self, args
    ):
        super(FCNet, self).__init__()
        self.is_deconv = True
        self.in_channels = 3
        self.is_batchnorm = True
        self.feature_scale = 4

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 11, 1, 5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, inputs):
        return self.conv(inputs)
