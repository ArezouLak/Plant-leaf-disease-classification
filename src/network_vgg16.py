import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU,MaxPool2d,Linear, Dropout


#vgg from scratch

class NetworkVGG16 (torch.nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.num_classes= num_classes
        self.activation= ReLU()

        # first block
        self.conv1= Conv2d(in_channels= 3, out_channels= 64, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN1= BatchNorm2d(64)

        self.conv2= Conv2d(in_channels= 64, out_channels= 64, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN2= BatchNorm2d(64)

        self.pool1= MaxPool2d(kernel_size=(2,2), stride=2)

        #second block

        self.conv3= Conv2d(in_channels= 64, out_channels= 128, kernel_size=(3,3),stride=1, padding=1)
       
        self.batchN3= BatchNorm2d(128)

        self.conv4= Conv2d(in_channels= 128, out_channels= 128, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN4= BatchNorm2d(128)

        self.pool2= MaxPool2d(kernel_size=(2,2), stride=2)

        #third block

        self.conv5= Conv2d(in_channels= 128, out_channels= 256, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN5= BatchNorm2d(256)

        self.conv6= Conv2d(in_channels= 256, out_channels= 256, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN6= BatchNorm2d(256)

        self.conv7= Conv2d(in_channels= 256, out_channels= 256, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN7= BatchNorm2d(256)


        self.pool3= MaxPool2d(kernel_size=(2,2), stride=2)


        #4th block

        self.conv8= Conv2d(in_channels= 256, out_channels= 512 ,kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN8= BatchNorm2d(512)

        self.conv9= Conv2d(in_channels= 512, out_channels= 512, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN9= BatchNorm2d(512)

        self.conv10= Conv2d(in_channels= 512, out_channels= 512, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN10= BatchNorm2d(512)


        self.pool4= MaxPool2d(kernel_size=(2,2), stride=2)


        #5th block

        self.conv11= Conv2d(in_channels= 512, out_channels= 512 ,kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN11= BatchNorm2d(512)

        self.conv12= Conv2d(in_channels= 512, out_channels= 512, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN12= BatchNorm2d(512)

        self.conv13= Conv2d(in_channels= 512, out_channels= 512, kernel_size=(3,3),stride=1, padding=1)
        
        self.batchN13= BatchNorm2d(512)


        self.pool5= MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1= Linear(in_features= 512 * 7 * 7, out_features=4096)
        self.do1=Dropout(0.5)
        self.fc2= Linear(in_features= 4096, out_features=4096)
        self.do2=Dropout(0.5)
        self.fc3= Linear(in_features=4096, out_features=self.num_classes)


    def forward (self,x):
        x = self.activation(self.batchN1(self.conv1(x)))
        x = self.activation(self.batchN2(self.conv2(x)))
        x = self.pool1(x)

        x = self.activation(self.batchN3(self.conv3(x)))
        x = self.activation(self.batchN4(self.conv4(x)))
        x = self.pool2(x)

        x = self.activation(self.batchN5(self.conv5(x)))
        x = self.activation(self.batchN6(self.conv6(x)))
        x = self.activation(self.batchN7(self.conv7(x)))
        x = self.pool3(x)

        x = self.activation(self.batchN8(self.conv8(x)))
        x = self.activation(self.batchN9(self.conv9(x)))
        x = self.activation(self.batchN10(self.conv10(x)))
        x = self.pool4(x)

        x = self.activation(self.batchN11(self.conv11(x)))
        x = self.activation(self.batchN12(self.conv12(x)))
        x = self.activation(self.batchN13(self.conv13(x)))
        x = self.pool5(x)

        # [batch, 512, 7, 7]
        x = torch.flatten(x, 1)
        # [batch, 25088]
        x=self.do1(self.activation(self.fc1(x)))
        x=self.do2(self.activation(self.fc2(x)))

        logits=self.fc3(x)

        return logits
    
    
        
        





        
        
          