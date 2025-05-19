import numpy as np
import warnings
import yaml
import os

config_path = 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
nb_class = config["model"]["num_classes"]


def PA(image, target) :
    mask = image == target
    well_classed = mask.sum().item()
    total = image.numel()
    return well_classed / total

def pixel_accuracy(images, targets) :
    res = []
    for image, target in zip(images, targets) :
        pa = PA(image, target)
        res.append(pa)
    return np.mean(res) 
    
def MPA(image, target) :
    mpa = []
    for i in range(0, nb_class) :
        mask_tar = target == i
        mask_im = image == i
        mask = (mask_tar & mask_im)
        well_classed = mask.sum().item() 
        total = mask_tar.sum().item()
        if total != 0 :
            mpa.append(well_classed / total)
        else :
            mpa.append(np.nan)            
    return mpa
 
def mean_pixel_accuracy_weighted(images, targets, weights) : 
    res = []
    for image, target in zip(images, targets) :
        mpa = MPA(image, target)
        res.append(mpa)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pa_by_class = np.nanmean(res, axis=0)
        
    return np.nansum(pa_by_class * weights) / np.nansum(weights)


def mean_pixel_accuracy(images, targets) : 
    res = []
    for image, target in zip(images, targets) :
        mpa = MPA(image, target)
        res.append(mpa)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pa_by_class = np.nanmean(res, axis=0)
        
    return np.nanmean(pa_by_class)


def IOU(image, target):
    iou_per_class = []

    for class_idx in range(0, nb_class):
        
        target_mask = target == class_idx
        prediction_mask = image == class_idx

        intersection = (target_mask & prediction_mask).sum().item()
        union = (target_mask | prediction_mask).sum().item()

        if union != 0:
            iou_per_class.append(intersection / union)
        else :
            iou_per_class.append(np.nan)

    return iou_per_class

def mean_iou(images, targets):
    iou_list = []

    for image, target in zip(images, targets):
        iou_per_image = IOU(image, target)
        iou_list.append(iou_per_image)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        iou_per_class = np.nanmean(iou_list, axis=0) 
        
    mean_iou = np.nanmean(iou_per_class)         
    return mean_iou

def mean_iou_weighted(images, targets, weights):
    iou_list = []

    for image, target in zip(images, targets):
        iou_per_image = IOU(image, target)
        iou_list.append(iou_per_image)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        iou_per_class = np.nanmean(iou_list, axis=0) 
        
    mean_iou = np.nansum(iou_per_class * weights) / np.nansum(weights)
             
    return mean_iou