import torch.nn as nn


class FineTuneSBP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsampling_factor=2):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=64),
            nn.Linear(in_features=1, out_features=1, bias=False),
        )

    def forward(self, x):
        return self.fine_tune(x)


class FineTuneDBP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsampling_factor=2):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=64),
            nn.Linear(in_features=1, out_features=1, bias=False),
        )

    def forward(self, x):
        return self.fine_tune(x)


class FineTuneMAP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsampling_factor=2):
        super().__init__()

        self.fine_tune = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=4,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(kernel_size=64),
            nn.Linear(in_features=1, out_features=1, bias=False),
        )

    def forward(self, x):
        return self.fine_tune(x)
