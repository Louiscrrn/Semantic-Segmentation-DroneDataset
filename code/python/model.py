import segmentation_models_pytorch as smp
from transformers import SegformerModel
import torch
import torch.nn as nn


# U-Net Models        
class Multi_Unet(nn.Module) :
    
    def __init__(self, num_classes=5, enc_name="resnet18", enc_in_channel=3, depth = 5, enc_weights="imagenet") :
        super().__init__()
        
        # Encodeur pré-entrainé
        self.encoder = smp.encoders.get_encoder(
            name=enc_name, 
            in_channels=enc_in_channel, 
            depth=depth, 
            weights=enc_weights)
        
        encoder_channels = self.encoder.out_channels
        
        # Décodeur à la main :
        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(depth, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(encoder_channels[i], encoder_channels[i - 1], kernel_size=2, stride=2)
            )
            
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(encoder_channels[i - 1] + encoder_channels[i - 1], encoder_channels[i - 1], kernel_size=3, padding=1),
                    nn.ReLU(),  # Ajout de ReLU ici
                )
            )

        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        enc_features = self.encoder(x)

        x = enc_features[-1]
        
        for i in range(len(self.upconvs)):

            x = self.upconvs[i](x)   
            x = torch.cat([x, enc_features[-(i + 2)]], dim=1)  
            x = self.convs[i](x)

        output = self.final_conv(x)
        
        return output


class UnetBinary(nn.Module):
    def __init__(self, nb_classes, enc_name="resnet34", enc_in_channels=3, enc_weights="imagenet"):
        """
        UNet avec encodeur pré-entraîné.
        
        Args:
            nb_classes (int): Nombre de classes de sortie.
            enc_name (str): Nom de l'encodeur pré-entraîné (par exemple, 'resnet34').
            enc_in_channels (int): Nombre de canaux d'entrée (3 pour RGB, 1 pour niveaux de gris).
            enc_weights (str): Poids pré-entraînés pour l'encodeur (par exemple, 'imagenet').
        """
        super(UnetBinary, self).__init__()
        
        # Initialisation de l'encodeur pré-entraîné
        self.encoder = smp.encoders.get_encoder(
            name=enc_name,
            in_channels=enc_in_channels,
            depth=5,
            weights=enc_weights
        )
        
        # Récupération des dimensions des sorties des différents niveaux de l'encodeur
        encoder_channels = self.encoder.out_channels
        
        # BOTTLENECK
        self.bottleneck_conv = nn.Conv2d(encoder_channels[-1], encoder_channels[-1], kernel_size=3, padding=1)
        
        # DECODER
        self.up_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        for i in range(len(encoder_channels) - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(encoder_channels[i], encoder_channels[i - 1], kernel_size=2, stride=2)
            )
            self.decoder_convs.append(
                nn.Sequential(
                    nn.Conv2d(encoder_channels[i - 1] * 2, encoder_channels[i - 1], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        
        # FINAL
        self.final_conv = nn.Conv2d(encoder_channels[0], nb_classes, kernel_size=1)
        
        self._nb_classes = nb_classes

    def forward(self, x):
        """
        Propagation avant du modèle.
        """
        # Encodage
        enc_features = self.encoder(x)  # Liste des caractéristiques de l'encodeur
        
        # Bottleneck
        x = torch.relu(self.bottleneck_conv(enc_features[-1]))
        
        # Décodage
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)  # Upsampling
            x = torch.cat([x, enc_features[-(i + 2)]], dim=1)  # Concaténation avec l'encodeur
            x = self.decoder_convs[i](x)  # Convolution
        
        # Sortie finale
        out = self.final_conv(x)
        
        if self._nb_classes <= 1:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=1)
        
        return out
    
    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        for param in self.encoder.parameters():
            param.requires_grad = False



# Transformers :
class SegFormerEncoder(nn.Module):
    def __init__(self, enc_name):
        super(SegFormerEncoder, self).__init__()    
        self.model = SegformerModel.from_pretrained(enc_name, output_hidden_states=True)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.hidden_states
    
class SegFormerDecoder(nn.Module):
    def __init__(self, n_classes) :
        super(SegFormerDecoder, self).__init__()
        
        # MLP Architecture :
        self.linear_c = nn.ModuleList([
            nn.Linear(in_features=32, out_features=256, bias=True),
            nn.Linear(in_features=64, out_features=256, bias=True),
            nn.Linear(in_features=160, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=256, bias=True)
        ])
        
        # Upsamplers
        self.upsamplers = nn.ModuleList([
            nn.Identity(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        ])
        
        # Fusion equivalent to a MLP
        self.conv_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.upconv_fin = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.activation = nn.ReLU()
        self.conv_2 = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, hidden_states):
        fusion = []
        for i, feature in enumerate(hidden_states) : 
            
            B, C, H, W = feature.shape
            proj = self.linear_c[i](feature.flatten(2).transpose(1, 2))  # Vectorize l'image
            proj = proj.transpose(1, 2).view(B, -1, H, W)
            
            proj = self.upsamplers[i](proj)

             
            fusion.append(proj)
            
        fused_tensor = torch.cat(fusion, dim=1)

        out = self.activation(self.conv_1(fused_tensor))
        
        out = self.upconv_fin(out)

        out = self.conv_2(out)
        return out

class SegFormer(nn.Module):
    def __init__(self, enc_name="nvidia/mit-b0", n_classes=5):
        super(SegFormer, self).__init__() 
        self.encoder = SegFormerEncoder(enc_name)
        self.decoder = SegFormerDecoder(n_classes)
    
    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        for param in self.encoder.parameters():
            param.requires_grad = False    
    
    def forward(self, x) :
        hidden_states = self.encoder(x)
        out = self.decoder(hidden_states)
        return out

# Binary segformer : Same architecture but with sigmoid activation
class BinarySegFormerEncoder(nn.Module):
    def __init__(self, enc_name):
        super(BinarySegFormerEncoder, self).__init__()    
        self.model = SegformerModel.from_pretrained(enc_name, output_hidden_states=True)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.hidden_states
    
class BinarySegFormerDecoder(nn.Module):
    def __init__(self, n_classes=2):  # Par défaut pour binaire
        super(BinarySegFormerDecoder, self).__init__()
        
        # MLP Architecture :
        self.linear_c = nn.ModuleList([
            nn.Linear(in_features=32, out_features=256, bias=True),
            nn.Linear(in_features=64, out_features=256, bias=True),
            nn.Linear(in_features=160, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=256, bias=True)
        ])
        
        # Upsamplers
        self.upsamplers = nn.ModuleList([
            nn.Identity(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        ])
        
        # Fusion equivalent to a MLP
        self.conv_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.upconv_fin = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.activation = nn.ReLU()
        self.conv_2 = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))  # 1 canal pour binaire
        
    def forward(self, hidden_states):
        fusion = []
        for i, feature in enumerate(hidden_states): 
            
            B, C, H, W = feature.shape
            proj = self.linear_c[i](feature.flatten(2).transpose(1, 2))  # Vectorize l'image
            proj = proj.transpose(1, 2).view(B, -1, H, W)
            
            proj = self.upsamplers[i](proj)

            fusion.append(proj)
            
        fused_tensor = torch.cat(fusion, dim=1)

        out = self.activation(self.conv_1(fused_tensor))
        
        out = self.upconv_fin(out)

        out = self.conv_2(out)
        out = torch.sigmoid(out)  # Activation sigmoïde pour binaire
        return out

class BinarySegFormer(nn.Module):
    def __init__(self, enc_name="nvidia/mit-b0", n_classes=2):  # Par défaut pour binaire
        super(BinarySegFormer, self).__init__() 
        self.encoder = SegFormerEncoder(enc_name)
        self.decoder = SegFormerDecoder(n_classes)
    
    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        for param in self.encoder.parameters():
            param.requires_grad = False    
    
    def forward(self, x):
        hidden_states = self.encoder(x)
        out = self.decoder(hidden_states)
        return out


# Final Model - Combined both SegFormer + UNet 
# Trained and tested on Drone Dataset

class UFormer(nn.Module):
    def __init__(self, alpha = 0.5, beta=0.5, trained = False, path="gricad/"):
        super(UFormer, self).__init__()
        
        self.UNet = Multi_Unet()
        self.SegFormer = SegFormer()
        
        self.alpha = alpha
        self.beta = beta
        
        if trained == True :
            self.UNet, self.SegFormer = self._load_models(path)    
    
    def _load_models(self, path) :
        Unet = Multi_Unet(num_classes=5, enc_weights="imagenet")
        SegFo = SegFormer(enc_name="nvidia/mit-b0", n_classes=5)
        Unet.load_state_dict(torch.load( path + "14_1_25/single_gpu/MultiUnet_3/MultiUnet_1000_14_25/" + "MultiUnet" + ".pt", map_location=torch.device('cuda'), weights_only=True))
        SegFo.load_state_dict(torch.load(path + "14_1_25/multi_gpu/SegFormer_3/SegFormer/" + "SegFormer" + ".pt", map_location=torch.device('cuda'), weights_only=True))
        return Unet, SegFo
    
    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        self.UNet.freeze_encoder()
        self.SegFormer.freeze_encoder()
    
    def forward(self, x) :
        
        mask_pred_SegFo = self.SegFormer(x)
        mask_pred_Unet = self.UNet(x)
        mask_res = self.alpha * mask_pred_SegFo + self.beta * mask_pred_Unet
        
        return mask_res

class BinaryUFormer(nn.Module):
    def __init__(self, alpha = 0.5, beta=0.5, trained = False, path=""):
        super(BinaryUFormer, self).__init__()
        
        self.UNet = UnetBinary(nb_classes=2)
        self.SegFormer = BinarySegFormer()
        
        self.alpha = alpha
        self.beta = beta
        
        if trained == True :
            self.UNet, self.SegFormer = self._load_models(path)    
    
    def _load_models(self, path) :
        Unet = UnetBinary(num_classes=2, enc_weights="imagenet")
        SegFo = BinarySegFormer(enc_name="nvidia/mit-b0", n_classes=2)
        Unet.load_state_dict(torch.load( path + "MultiUnet" + ".pt", map_location=torch.device('cuda'), weights_only=True))
        SegFo.load_state_dict(torch.load(path + "SegFormer" + ".pt", map_location=torch.device('cuda'), weights_only=True))
        return Unet, SegFo
    
    def freeze_encoder(self):
        """Geler les poids de l'encodeur pour ne pas les entraîner."""
        self.UNet.freeze_encoder()
        self.SegFormer.freeze_encoder()
    
    def forward(self, x) :
        
        mask_pred_SegFo = self.SegFormer(x)
        mask_pred_Unet = self.UNet(x)
        mask_res = self.alpha * mask_pred_SegFo + self.beta * mask_pred_Unet
        
        return mask_res

# Utilitaires :

def count_parameters(model):
    """Retourne le nombre total de paramètres et ceux qui sont entraînables."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params
