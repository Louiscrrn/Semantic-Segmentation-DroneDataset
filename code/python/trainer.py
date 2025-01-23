import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from python.vizualization import * 
from tqdm import tqdm
from python.metrics import *


class Trainer :
    """Classe qui encapsule les méthodes utiles à l'entrainement, validation et test d'un modèle.
    """

    # Méthodes publiques
    def __init__(self, model, yaml_config, train_dataset, val_dataset, test_dataset, criterion, optimizer, scheduler, device) :
        
        self.config = yaml_config
        self.distributed = yaml_config['distributed']['active']
        self.output_path = yaml_config['training']['output_path'] + yaml_config['model']['type'] + '/'
        
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if self.distributed == 1 else None
        self.model = model.to(self.gpu_id)
        
        batch_size = yaml_config['training']['batch_size']
        num_workers = yaml_config['training']['num_workers']
        pin_memory = True if self.distributed == 1 else False
        
        shuffle = False if self.distributed == 1 else True
        train_sampler = DistributedSampler(train_dataset) if self.distributed == 1 else None
        val_sampler = DistributedSampler(val_dataset) if self.distributed == 1 else None
        test_sampler = DistributedSampler(test_dataset) if self.distributed == 1 else None
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, sampler=train_sampler, pin_memory=pin_memory) if train_dataset else None
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, sampler=val_sampler, pin_memory=pin_memory) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, sampler=test_sampler, pin_memory=pin_memory) if test_dataset else None
        
        self.test_dataset = test_dataset
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.device = device
        
        self.model = DistributedDataParallel(self.model, device_ids=[self.gpu_id]) if self.distributed == 1 else model.to(device)
        self.model.training = True
    
    def fit(self, max_epochs: int):
        
        start_time = time.time()
        valid_loss_min = np.Inf
        data = []
        
        for epoch in range(max_epochs):        
            valid_loss_min, data = self._run_epoch(epoch, valid_loss_min, data)
            
        total_time = time.time() - start_time
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best Validation Loss: {valid_loss_min:.6f}")
        
        return pd.DataFrame(data)

    def predict(self, test_data_idxs) :
        self.model.eval()
        with torch.no_grad() :
            for test_data_idx in test_data_idxs :
                img_test, mask_test = self.test_dataset.__getitem__(test_data_idx)
                img_test, mask_test = img_test.to(self.device), mask_test.to(self.device)
                img_test = img_test.unsqueeze(0)
                mask_test = mask_test.unsqueeze(0)
                out = self.model(img_test)
                iou_w, iou, mpa_w, mpa = self._get_metrics(out, mask_test)
                print(f"IOU weighted : {iou_w:.4f} | Iou : {iou:.4f} ")
                print(f"MPA weighted : {mpa_w:.4f} | MPA : {mpa:.4f}")
                    
                fig = self._plot_pred(img_test, mask_test, out)                
                plt.suptitle(f"Prediction n°{test_data_idx}")
                if self.output_path :
                    plt.savefig(self.output_path + self.config["model"]["type"] + f"_pred_{test_data_idx}.png")

    def test(self)  :
        print(f"######### TESTING #########" )
        data = []
        t_loss, t_iou, t_iou_w, t_mpa, t_mpa_w = self._evaluate(self.test_loader)
        print(f"Test Loss: {t_loss:.4f}")
        print(f"Test IoU: {t_iou:.4f} | Test IoU weighted: {t_iou_w:.4f} ")
        print(f"Test MPA: {t_mpa:.4f} | Test MPA weighted: {t_mpa_w:.4f}")
        data.append({
            "test_loss": t_loss,
            "test_iou": t_iou,
            "test_iou_w" : t_iou_w,
            "test_mpa": t_mpa,
            "test_mpa_w" : t_mpa_w,
        })
        return pd.DataFrame(data)
  
    # Méthodes privées :
    def _get_metrics(self, outputs, masks) : 
        outputs = torch.argmax(outputs, 1).unsqueeze(1)
        weights = [round(i, 3) for i in self._class_weights(masks)]
        
        iou = mean_iou(outputs, masks)
        iou_w = mean_iou_weighted(outputs, masks, weights)
        mpa_w = mean_pixel_accuracy_weighted(outputs, masks, weights)
        mpa = mean_pixel_accuracy(outputs, masks)
        
        return iou_w, iou, mpa_w, mpa 
            
    def _class_weights(self, masks):
        num_classes = self.config['model']['num_classes']
        class_counts = torch.zeros(num_classes, device=self.device)
        n, m = masks.size(2),  masks.size(3)
        
        for class_idx in range(num_classes):
            class_counts[class_idx] = (masks == class_idx).sum().item()
            class_counts[class_idx] = class_counts[class_idx] / (n * m)   
 
        return  class_counts.cpu().tolist()
        
    def _run_batch(self, imgs, masks):
        self.optimizer.zero_grad()
        outputs = self.model(imgs)
        loss = self.criterion(outputs, masks)
        
        if self.model.training:
            loss.backward()
            self.optimizer.step()
            
        loss = loss.item()
        iou_w, iou, mpa_w, mpa = self._get_metrics(outputs, masks)
        
        return loss, iou_w, iou, mpa_w, mpa

    def _run_epoch(self, epoch, valid_loss_min, data):
        
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        
        if self.config['distributed']['active'] == 1 :
            self.train_loader.sampler.set_epoch(epoch)
        
        t_loss, t_iou, t_iou_w, t_mpa, t_mpa_w = self._train() 
        v_loss, v_iou, v_iou_w, v_mpa, v_mpa_w = self._evaluate(self.val_loader)
        
        self._print_metrics(t_loss, t_iou, t_iou_w, t_mpa, t_mpa_w, v_loss, v_iou, v_iou_w, v_mpa, v_mpa_w)
        valid_loss_min = self._next(v_loss, valid_loss_min)
        
        data.append({
            "epoch": epoch + 1,
            "train_loss": t_loss,
            "val_loss": v_loss,
            "train_iou": t_iou,
            "val_iou": v_iou,
            "train_iou_w": t_iou_w,
            "val_iou_w": v_iou_w,
            "train_mpa": t_mpa,
            "val_mpa": v_mpa,
            "train_mpa_w": t_mpa_w,
            "val_mpa_w": v_mpa_w,
        })
        
        return valid_loss_min, data
        
    def _next(self, v_loss, valid_loss_min) :
        if self.distributed != 1 :   
            if v_loss < valid_loss_min - self.config['training']['tolerance']:
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> {v_loss:.6f}). Saving model...")
                output_dir = os.path.dirname(self.output_path + self.config['model']['type'] + '.pt')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(self.model.state_dict(), self.output_path + self.config['model']['type'] + '.pt')
                valid_loss_min = v_loss
            if self.scheduler is not None:
                self.scheduler.step(v_loss)
                for i, param_group in enumerate(self.optimizer.state_dict()['param_groups']):
                    print(f"Current learning rate (group {i}): {param_group['lr']:.6e}")
   
        else :
            if v_loss < valid_loss_min - self.config['training']['tolerance'] and self.gpu_id == 0 :
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> {v_loss:.6f}). Saving model...")
                torch.save(self.model.module.state_dict(), self.output_path + self.config['model']['type'] + '.pt')
                valid_loss_min = v_loss
            if self.scheduler is not None and self.gpu_id == 0:
                self.scheduler.step(v_loss)
                for i, param_group in enumerate(self.optimizer.state_dict()['param_groups']):
                    print(f"Current learning rate (group {i}): {param_group['lr']:.6e}")
        print(f"==========================================================================================")
        return valid_loss_min
        
    def _print_metrics(self, t_loss, t_iou, t_iou_w, t_mpa, t_mpa_w, v_loss, v_iou, v_iou_w, v_mpa, v_mpa_w):
        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"Train IoU: {t_iou:.4f} | Val IoU: {v_iou:.4f}")
        print(f"Train IoU weighted : {t_iou_w:.4f} | Val IoU weighted : {v_iou_w:.4f}")
        print(f"Train MPA: {t_mpa:.4f} | Val MPA: {v_mpa:.4f}")
        print(f"Train MPA weighted : {t_mpa_w:.4f} | Val MPA weighted : {v_mpa_w:.4f}")
                
    def _train(self) : 
        t_loss = 0.0 
        t_iou = 0.0 
        t_iou_w = 0.0
        t_mpa = 0.0
        t_mpa_w = 0.0
        self.model.train()
        self.model.training = True
        for images, masks in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device) if self.distributed == 0 else images.to(self.gpu_id)
            masks = masks.to(self.device) if self.distributed == 0 else masks.to(self.gpu_id)
            loss, iou_w, iou, mpa_w, mpa = self._run_batch(images, masks.long())
            t_loss += loss
            t_iou += iou
            t_iou_w += iou_w
            t_mpa += mpa
            t_mpa_w += mpa_w
            if self.device == 'cuda' : torch.cuda.empty_cache()
        return t_loss / len(self.train_loader), t_iou / len(self.train_loader), t_iou_w / len(self.train_loader), t_mpa / len(self.train_loader), t_mpa_w / len(self.train_loader)
    
    def _evaluate(self, loader) : 
        t_loss = 0.0
        t_iou = 0.0
        t_iou_w = 0.0
        t_mpa = 0.0
        t_mpa_w = 0.0
        self.model.eval()
        self.model.training = False
        with torch.no_grad() :
            for images, masks in tqdm(loader, desc="Evaluating", leave=False):
                images = images.to(self.device) if self.distributed == 0 else images.to(self.gpu_id)
                masks = masks.to(self.device) if self.distributed == 0 else masks.to(self.gpu_id)
                loss, iou_w, iou, mpa_w, mpa = self._run_batch(images, masks.long())
                t_loss += loss
                t_iou += iou
                t_iou_w += iou_w
                t_mpa += mpa
                t_mpa_w += mpa_w
                if self.device == 'cuda' : torch.cuda.empty_cache()
            return t_loss / len(loader), t_iou / len(loader), t_iou_w / len(loader), t_mpa / len(loader), t_mpa_w / len(loader)
    
    def _plot_pred(self, img_test, mask_test, predicted_mask) : 
            
            img_test = img_test.squeeze(0)
            mask_test = mask_test.squeeze(0).squeeze(0)
            predicted_mask = torch.argmax(predicted_mask, 1).squeeze(0)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
            axes[0].imshow(img_test.cpu().squeeze().permute(1, 2, 0).numpy())  # Image originale
            axes[0].set_title("Original Image")
                
            axes[1] = get_axes_mask(mask_test.cpu(), cmap=COLOR_MAP, axes=axes[1])
            axes[1].set_title("Ground Truth")
                    
            axes[2] = get_axes_mask(predicted_mask.cpu(), cmap=COLOR_MAP, axes=axes[2])
            axes[2].set_title("Predicted Mask")
            
            return fig