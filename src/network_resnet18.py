import torch
from torch.nn.modules import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d,ReLU, BatchNorm2d,AvgPool2d, Linear


class NetworkResNet18(Module):

    def __init__(self, num_classes):

        super().__init__()

        self.num_classes=num_classes
        
        #first block

        self.conv1= Conv2d(3, 64, kernel_size=(7,7), stride=2, padding=3)
        self.relu= ReLU()
        self.bn1= BatchNorm2d(64)

        self.pool1=MaxPool2d(kernel_size=(3,3), stride=2, padding=1)

        #second block

        self.conv2= Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.bn2=BatchNorm2d(64)
        self.conv3= Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.bn3=BatchNorm2d(64)
        self.conv4= Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.bn4=BatchNorm2d(64)
        self.conv5= Conv2d(64, 64, (3,3), stride=1, padding=1)
        self.bn5=BatchNorm2d(64)

        #third block

        self.conv6= Conv2d(64, 128, (3,3),stride=2, padding=1)
        self.bn6=BatchNorm2d(128)
        self.conv7= Conv2d(128, 128, (3,3),stride=1, padding=1)
        self.bn7=BatchNorm2d(128)
        self.conv8= Conv2d(128, 128, (3,3),stride=1, padding=1)
        self.bn8=BatchNorm2d(128)
        self.conv9= Conv2d(128, 128, (3,3),stride=1, padding=1)
        self.bn9=BatchNorm2d(128)

        #4th block
        self.conv10= Conv2d(128, 256, (3,3),stride=2, padding=1)
        self.bn10=BatchNorm2d(256)
        self.conv11= Conv2d(256, 256, (3,3),stride=1, padding=1)
        self.bn11=BatchNorm2d(256)
        self.conv12= Conv2d(256, 256, (3,3),stride=1, padding=1)
        self.bn12=BatchNorm2d(256)
        self.conv13= Conv2d(256, 256, (3,3),stride=1, padding=1)
        self.bn13=BatchNorm2d(256)

        #5th block
        self.conv14= Conv2d(256, 512, (3,3),stride=2, padding=1)
        self.bn14=BatchNorm2d(512)
        self.conv15= Conv2d(512, 512, (3,3),stride=1, padding=1)
        self.bn15=BatchNorm2d(512)
        self.conv16= Conv2d(512, 512, (3,3),stride=1, padding=1)
        self.bn16=BatchNorm2d(512)
        self.conv17= Conv2d(512, 512, (3,3),stride=1, padding=1)
        self.bn17=BatchNorm2d(512)

        self.pool2=AvgPool2d((7,7))
        self.fc= Linear(in_features=512, out_features=num_classes)

        #three skip connections
        self.skip3= torch.nn.Sequential(Conv2d(64,128, (1,1), stride=2), BatchNorm2d(128))
        self.skip5=torch.nn.Sequential(Conv2d(128, 256, (1,1) , stride=2), BatchNorm2d(256))
        self.skip7=torch.nn.Sequential(Conv2d(256, 512, (1,1), stride=2), BatchNorm2d(512))

        


        
    def forward(self,x):

        x= self.relu(self.bn1(self.conv1(x)))
        x=self.pool1(x)
        skip1=x
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x= x +skip1

        x=self.relu(x)
        skip2=x


        x=self.relu(self.bn4(self.conv4(x)))
        x=self.bn5(self.conv5(x))
        x=skip2+ x

        x=self.relu(x)
        skip3=self.skip3(x)
        
        x=self.relu(self.bn6(self.conv6(x)))
        x=self.bn7(self.conv7(x))

        x= skip3 + x

        x=self.relu(x)

        skip4=x

        x=self.relu(self.bn8(self.conv8(x)))
        x=self.bn9(self.conv9(x))
        x=skip4 + x

        x= self.relu(x)

        skip5=self.skip5(x)

        x= self.relu(self.bn10(self.conv10(x)))

        x=self.bn11(self.conv11(x))

        x= skip5 +x     

        x=self.relu(x)

        skip6=x

        x=self.relu(self.bn12(self.conv12(x)))

        x=self.bn13(self.conv13(x))

        x=skip6 +x

        x=self.relu(x)

        skip7=self.skip7(x)
        
        x=self.relu(self.bn14(self.conv14(x)))
        x=self.bn15(self.conv15(x))
        x=skip7 +x
        
        x=self.relu(x)
        skip8=x

        x=self.relu(self.bn16(self.conv16(x)))
        x=self.bn17(self.conv17(x))

        x= skip8 + x
        x=self.relu(x)
        x=self.pool2(x)
        x=torch.flatten(x, 1)
        logits=self.fc(x)

        return logits






# x
# │
# ├────────────────────────┐
# │                        │
# ↓                        │
# Conv → BN → ReLU         │
# ↓                        │
# Conv → BN                │
# │                        │
# └────── Add ←────────────┘
#          ↓
#         ReLU








        
