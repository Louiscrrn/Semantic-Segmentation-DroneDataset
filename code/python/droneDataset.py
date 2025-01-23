from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import os
from PIL import Image
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.transforms import InterpolationMode
import yaml

TEST_SIZE = 50

config_path =  'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

if config["model"]["num_classes"] == 2:
    COLOR_MAP = {
        0: ([155, 38, 182], 'Obstacles'), 
        1: ([14, 135, 204], 'Background')
    }
else:
    COLOR_MAP = {
        0: ([155, 38, 182], 'Obstacles'),   
        1: ([14, 135, 204], 'Water'),   
        2: ([124, 252, 0], 'Nature'),     
        3: ([255, 20, 147], 'Moving'),     
        4: ([169, 169, 169], 'Landable'),     
    }



# Classe Dataset :

class DroneDataset(torch.utils.data.Dataset) :
    """
    Classe DroneDataset, permet de charger les données en mémoire 
    pour ensuite les manipuler avec PyTorch. 
    Voir la docu ou notebook playground.ipynb
    """
    def __init__(self, 
                 
                 data_path : Path,
                 img_folder : str,
                 mask_folder : str, 
                
                 data,
                 transform,
                 augmentation,
                 
                 type : str = "TRAIN", 
                
                 segmentation_mode : str = "multi",
                 labels = COLOR_MAP,
                
                 ) :
    
        self._data_path = data_path
        self._imgs_folder = img_folder
        self._masks_folder = mask_folder
        
        self._imgs = sorted(os.listdir( data_path +  img_folder ))
        self._masks = sorted(os.listdir( data_path + mask_folder ))
        
        self.cfg_data = data
        self.cfg_transform = transform
        self.cfg_augmentation = augmentation
        
        self.type = type
        
        size = data['size'] - TEST_SIZE
        train_size = int(size * data['train'])
        val_size = int(size * data['val'])
        test_size = TEST_SIZE
        
        if type == "TRAIN" :
            self._imgs = self._imgs[:train_size]
            self._masks = self._masks[:train_size]
        
        if type == "VAL" :
            self._imgs = self._imgs[train_size:train_size+val_size]
            self._masks = self._masks[train_size:train_size+val_size]
        
        if type == "TEST" : 
            self._imgs = self._imgs[-TEST_SIZE:]
            self._masks = self._masks[-TEST_SIZE:]
            #self._imgs = self._imgs[train_size+val_size:train_size+val_size+test_size]
            #self._masks = self._masks[train_size+val_size:train_size+val_size+test_size]
            
        self._segmentation_mode = segmentation_mode
        self._cmap = labels
           
    def __getitem__(self, idx: int) :
                
        if idx < len(self._imgs) :
            img =  Image.open(self._data_path + '/' + self._imgs_folder + '/' +  self._imgs[idx])
            mask =  Image.open(self._data_path + '/' + self._masks_folder + '/' +  self._masks[idx])
            img, mask =  self._resize(img, mask)
        else :
            idx = torch.randint(0, len(self._imgs), (1,)).item()
            img =  Image.open(self._data_path + '/' + self._imgs_folder + '/' +  self._imgs[idx])
            mask =  Image.open(self._data_path + '/' + self._masks_folder + '/' +  self._masks[idx])
            img, mask = self._resize(img, mask)
            img, mask = self._augment(img, mask)
            
        mask = self._compute_mask(mask)
        return img, mask
            
    def _compute_mask(self, mask) :
        if self.cfg_data['segmentation_mode'] == 'binary' :
            mask = (mask > 0.0).float()
            return mask
        elif self.cfg_data['segmentation_mode'] == 'multi' :
            mask = raw_to_class((mask * 255).long())
            return mask
                
    def _resize(self, img, mask) :
        if self.cfg_transform['active'] == 1 :
            transform = transforms.Resize((self.cfg_transform['width'], self.cfg_transform['height']))
            img = F.to_tensor(transform(img)) 
            mask = F.to_tensor(transform(mask))
            return img, mask
        else :
            return F.to_tensor(img), F.to_tensor(mask)  
    
    def _augment(self, img, mask) : 
        transfo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=self.cfg_augmentation['RandomRotation']),
            transforms.CenterCrop(size=(int(img.shape[1] * self.cfg_augmentation['zoom_in']), int(img.shape[2]* self.cfg_augmentation['zoom_in']))),
            transforms.Resize(size=(img.shape[1], img.shape[2]), interpolation=InterpolationMode.NEAREST),
            ])
        
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        img = transfo(img)
        torch.manual_seed(seed)
        mask = transfo(mask)
        return img, mask
               
    def __len__(self) :
        """Len du jeu de données

        Returns:
            int: taille de la liste les noms des fichiers
        """
        assert( len(self._imgs) == len(self._masks) )
        
        if self.type == "TEST" :
            return len(self._imgs)
        
        if self.type == "TRAIN" :
            augmentation_size = int(self.cfg_augmentation['size'] * self.cfg_data['train']) if self.cfg_augmentation['active'] == 1 else 0
        
        if self.type == "VAL" :
            augmentation_size = int(self.cfg_augmentation['size'] * self.cfg_data['val']) if self.cfg_augmentation['active'] == 1 else 0
        
        return len(self._imgs) + augmentation_size
    
    def print(self, idx : int):
        """Affiche l'échantillons donné par son index

        Args:
            idx (int): index de l'échantillon souhaité
        """
        image_tensor, mask_tensor = self.__getitem__(idx)

        image = image_tensor.permute(1, 2, 0).cpu().numpy()  
        mask = class_to_rgb(mask_tensor)
                   

        if self._segmentation_mode == 'binary' :
            mask = mask.squeeze()        
   
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(mask)
        axes[1].set_title("Binary Segmented Mask")
        if self._segmentation_mode == 'multi':
            legend_elements = [
                Patch(facecolor=np.array(color) / 255, label=label)
                for color, label in self._cmap.values()
            ]
            axes[1].legend(handles=legend_elements, loc='best', title="Classes")
        
        plt.suptitle(f"Sample n°{idx}")
        plt.tight_layout()
        plt.show()
        
    def printra(self, nb : int):
        
        random_indices = random.sample(range(self.__len__()), nb)
        
        for idx in random_indices :
            self.print(idx)

# Manipulation des données :

def rgb_to_class(mask_rgb, cmap = COLOR_MAP):
    h, w, _ = mask_rgb.shape
    class_mask = torch.zeros((h, w), dtype=torch.long, device=mask_rgb.device)
    
    for class_idx, (color, _) in cmap.items():
        color_tensor = torch.tensor(color, dtype=torch.uint8, device=mask_rgb.device)
        match = torch.all(mask_rgb == color_tensor, dim=-1)
        class_mask[match] = class_idx
    
    return class_mask.unsqueeze(0)

def raw_to_rgb(mask, cmap=COLOR_MAP):
    if mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.squeeze(0)
    
    n, m = mask.shape
    
    image = torch.zeros((n, m, 3), dtype=torch.uint8, device=mask.device)

    for key, (color, _) in cmap.items():
        color_tensor = torch.tensor(color, dtype=torch.uint8, device=mask.device)
        image[mask == key] = color_tensor
    
    return image

def class_to_rgb(class_mask, cmap=COLOR_MAP):

    if class_mask.dim() == 3 and class_mask.size(0) == 1:
        class_mask = class_mask.squeeze(0)

    h, w = class_mask.shape

    rgb_image = torch.zeros((h, w, 3), dtype=torch.uint8, device=class_mask.device)

    for class_idx, (color, _) in cmap.items():
        color_tensor = torch.tensor(color, dtype=torch.uint8, device=class_mask.device)
        rgb_image[class_mask == class_idx] = color_tensor

    return rgb_image

def raw_to_class(mask, cmap=COLOR_MAP):
    return rgb_to_class(raw_to_rgb(mask))
