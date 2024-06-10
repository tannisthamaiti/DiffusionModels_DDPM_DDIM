import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from diffusion_utils import *
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML



def get_backbone(name, pretrained=True):

    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones
     # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(weights=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(weights=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_onebefore_output = 'layer3'
        backbone_output = 'layer4'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone.to(device), feature_names, backbone_output, backbone_onebefore_output


class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out
         # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)
        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None
    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)
        return x
class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,in_channels,n_feat, c_feat=None,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        self.backbone_name = backbone_name
        self.n_feat = n_feat
        self.n_cfeat = 5
        self.in_channels= in_channels
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.backbone, self.shortcut_features, self.bb_out_name, self.bb_onebefore_out_name = get_backbone(backbone_name, pretrained=pretrained)
        
        

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        
        num_blocks = len(self.shortcut_features)
        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.up0 = nn.ConvTranspose2d(in_channels=2*n_feat, out_channels=2*n_feat, kernel_size=(8, 8),
                                         stride=8, padding=0, output_padding=0)
        self.up00 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False)
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((1)), nn.GELU())
        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )
        
        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))


        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input,t=None, c=None):

        """ Forward propagation in U-Net. """

        x_init = self.init_conv(input)
        x, features = self.forward_backbone(input)
        down1,features_layer3= self.forward_onebefore_backbone(input)

        #print(f"input: {input.shape}")
        # hiddenvec
        hiddenvec = self.to_vec(x)
        down2=self.up0(hiddenvec)
        down1=self.up00(down1)

        # embed context and timestep
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"down1: {down1.shape}")
        up1 = self.up0(hiddenvec)
        cemb1= torch.empty(batch_size, 512, 1, 1).normal_().to(device)
        cemb2= torch.empty(batch_size, 256, 1, 1).normal_().to(device)
        #down1: torch.Size([32, 256, 64, 64])
        print(up1.shape, down2.shape, cemb1.shape, temb1.shape)
        up2 = self.up1(cemb1*up1+temb1,down2)
  
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x_init), 1))
        return out


    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features
    def forward_onebefore_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_onebefore_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param









# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 256 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 128 # 16x16 image
save_dir = 'weights/'

# training hyperparameters
batch_size = 32
n_epoch = 500
lrate=1e-5

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1


# construct model
nn_model = Unet(3,n_feat,backbone_name='resnet34').to(device)

dataset = CustomDataset("./wind_366X366.npy", "./wind_label_366X366.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
optim = torch.optim.AdamW(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

# training without context code

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model,epoch,optimizer

#load from checkpoint
nn_model.load_state_dict(torch.load(f"weights/lr1e-4/model_backbone_100.pth", map_location=device))
optim.load_state_dict(torch.load(f"weights/lr1e-4/model_backbone_optimizer100.pth", map_location=device))
# set into train mode
nn_model.train()
train_loss_epochs=[]
for ep in range(n_epoch): # modified epoch 
    print(f'epoch {ep}')
    
    # linearly decay learning rate
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    
    pbar = tqdm(dataloader, mininterval=2 )
    train_loss=[]
    for x, _ in pbar:   # x: images
        optim.zero_grad()
        
        x = x.float().to(device)
        if x.size(0)!=batch_size:
            new_size = batch_size - x.size(0)
            padding = torch.zeros(new_size, 3, 128, 128, device=device)
            x= torch.cat((x, padding), dim=0)
            
        
        
        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
        x_pert = perturb_input(x, t, noise)
        
        # use network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps)
        #pred_noise = nn_model(x_pert)
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        train_loss.append(loss.item())
        loss.backward()
        
        optim.step()

    # save model periodically
    if ep%4==0 or ep == int(n_epoch-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"model_backbone_{ep}.pth")
        torch.save(optim.state_dict(), save_dir + f"model_backbone_optimizer{ep}.pth")
        np.save(save_dir+f"loss_backbone_{ep}.npy", train_loss)
        print('saved model at ' + save_dir + f"model_backbone_{ep}.pth")
    train_loss_epochs.append(train_loss)
np.save("loss128X128.npy", train_loss_epochs)