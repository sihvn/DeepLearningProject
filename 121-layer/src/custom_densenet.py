## import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
# device = "cpu"
device = torch.device("cuda" if CUDA else "cpu")

torch.manual_seed(42)

class dense_layer(nn.Module):
    def __init__(self, dim, training):
        super(dense_layer, self).__init__()
        eps = 1e-5
        momentum = 0.1
        hidden_dim = 128
        output_dim = 32
        self.training = training
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=dim, eps=eps, momentum=momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=(1,1), stride=(1,1),bias=False),
            nn.BatchNorm2d(num_features=hidden_dim, eps=eps, momentum=momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        )
        
    def forward(self, x):
        dropout_rate = 0.2
        return F.dropout(self.net(x), p = dropout_rate, training=self.training)  

class dense_block(nn.ModuleDict):
    def __init__(self, layer_count, dim, training):
        super(dense_block, self).__init__()
        hidden_dim = dim
        growth_rate = 32
        for layer_index in range(layer_count):
            # setattr(self, f'layer_{layer_index}',dense_layer(hidden_dim, training=training))
            # ## update the hidden_dim so that the input size of the next layer has space
            # ## for all the outputs that precede it
            # hidden_dim += getattr(self,f'layer_{layer_index}').net[5].out_channels
            layer = dense_layer(
                dim = hidden_dim,
                training=training
            )
            self.add_module(f"dense_layer{layer_index}", layer)
            hidden_dim += growth_rate


    def forward(self, input):
        output = input
        for _, layer in self.items():
            # print("working in " + str(layer._get_name()) + " ...")
            new_output = layer(output)
            # print("output of this dense_layer is " + str(new_output.shape))
            ## add the new_output to the previous output
            ## for the next layer in the block
            # print("output shape", str(output.shape))
            # print("new_output shape", str(new_output.shape))
            output = torch.cat((output,new_output),1) 
        return output   
    
class transition_block(nn.Module):
    def __init__(self, channels):
        super(transition_block, self).__init__()
        eps = 1e-5
        momentum = 0.1
        self.norm = nn.BatchNorm2d(channels, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channels, channels//2, kernel_size=(1,1), stride=(1,1), bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self, input):
        x = input
        for layer in self.children():
            # print("working in " + str(layer._get_name()) + " ...")
            x = layer(x)
            # print("output of this transition is " + str(x.shape))

        return x

class dense_net(nn.Module):
    def __init__(self, num_class, training):
        super(dense_net, self).__init__()
        hidden_dim = 64

        self.initial_setup = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            nn.BatchNorm2d(num_features=hidden_dim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride = 2,padding=1, dilation=1, ceil_mode=False)
        )

        ## just calculate the number of channels in trans blocks manually
        ## since we know initial channel number and how many layers are in each block
        self.denseblock1 = dense_block(layer_count=6, dim=hidden_dim, training=training)
        self.trans1 = transition_block(channels = hidden_dim + 6 * 32) # 256 
        self.denseblock2 = dense_block(layer_count=12, dim=hidden_dim * 2, training=training)
        self.trans2 = transition_block(channels = hidden_dim * 2 + 12 * 32) #512
        self.denseblock3 = dense_block(layer_count=24, dim=hidden_dim * 4, training=training)
        self.trans3 = transition_block(channels = hidden_dim * 4 + 24 * 32) # 1024
        self.denseblock4 = dense_block(layer_count=16, dim=hidden_dim * 8, training=training)
        self.norm5 = nn.BatchNorm2d(num_features=hidden_dim * 8 + 16 * 32)

        ## preparing input into the classifier
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1)) ## global average pooling
        self.flatten = nn.Flatten()

        ## classifier component:
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.norm5.num_features, out_features=num_class, bias=True),
            nn.Sigmoid()
        )

        ## initialize to random values
        kaiming = norm = linear = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                kaiming = True
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                norm = True
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                linear = True

        # print(kaiming,norm,linear)

    def forward(self, input):
        x = input
        for block in self.children():
            # print("working in " + str(block._get_name()) + " ...")
            x = block(x)
            # print("output of this block is " + str(x.shape))
        return x