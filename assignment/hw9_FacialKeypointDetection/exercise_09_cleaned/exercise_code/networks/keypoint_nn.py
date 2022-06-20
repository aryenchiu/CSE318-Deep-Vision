"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules.pooling import MaxPool2d

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),     # [16, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),         # [16, 48, 48]

            nn.Conv2d(16, 32, 3, 1, 1),    # [32, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),         # [32, 24, 24]

            nn.Conv2d(32, 64, 3, 1, 1),    # [64, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),         # [64, 12, 12]

            nn.Conv2d(64, 128, 3, 1, 1),   # [128, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),         # [128, 6, 6]
            nn.Dropout2d(0.5),

            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        )

        self.cnn.apply(initialize_weights)
        self.fc.apply(initialize_weights)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        out = self.cnn(x)
        x = self.fc(out)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
