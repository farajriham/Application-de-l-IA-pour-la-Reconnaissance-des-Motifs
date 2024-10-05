from torch import nn
from torchvision.models import resnet18


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SimpleDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        #take inspiration from above without dropouts
        self.bbox_regressor = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 16) ,
            nn.ReLU(),
            nn.Linear(16,4),
            nn.Sigmoid()
        )
        self.bbox_regressor.apply(init_weights)

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        
        # Prédiction des classes
        class_predictions = self.classifier(x)
        
        # Prédiction des coordonnées de la boîte englobante
        bbox_predictions = self.bbox_regressor(x)
        
        return class_predictions, bbox_predictions




class DeeperDetector(nn.Module):
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 6, kernel_size=(5, 5), padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 12, kernel_size=(4,4 ), padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(12, 24, kernel_size=(4, 4), padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(24, 48, kernel_size=(5, 5), padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(48, 192, kernel_size=(1, 1), padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(192 * 3 * 3, 240),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(120, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        #take inspiration from above without dropouts
        self.bbox_regressor = nn.Sequential(
            nn.Linear(192 * 3 * 3, 240),
            nn.ReLU(),
            nn.Linear(240, 60) ,
            nn.ReLU(),
            nn.Linear(60,4),
            nn.Sigmoid()
        )
        self.bbox_regressor.apply(init_weights)

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        
        # Prédiction des classes
        class_predictions = self.classifier(x)
        
        # Prédiction des coordonnées de la boîte englobante
        bbox_predictions = self.bbox_regressor(x)
        
        return class_predictions, bbox_predictions


class ResnetObjectDetector(nn.Module):
    """ Resnet18 based feature extraction layers """
    def __init__(self, nb_classes):
        super().__init__()
        # copy resnet up to the last conv layer prior to fc layers, and flatten
        features = list(resnet18(pretrained=True).children())[:9]
        self.features = nn.Sequential(*features, nn.Flatten())

        # freeze all ResNet18 layers during the training process
        for param in self.features.parameters():
            param.requires_grad = False

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, nb_classes)
        )

        # create regressor path for bounding box coordinates prediction
        # take inspiration from above without dropouts
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512) ,
            nn.ReLU(),
            nn.Linear(512,4),
            nn.Sigmoid()
        )
        self.bbox_regressor.apply(init_weights)

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        x = self.features(x)
        class_predictions = self.classifier(x)
        bbox_predictions = self.bbox_regressor(x)

        #compute and add the bounding box regressor term
        return class_predictions, bbox_predictions
