import torch
import yaml
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from python.model import *  
from python.droneDataset import *  
from python.trainer import *
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

def init_torch_env(cfg_torch) :
    if cfg_torch['seed'] != 'None' :
        torch.manual_seed(cfg_torch['seed'])
        np.random.seed(cfg_torch['seed'])
        random.seed(cfg_torch['seed'])

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg_torch['seed'])
            torch.cuda.manual_seed_all(cfg_torch['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def get_dataset(cfg_data, cfg_transform, cfg_augmentation) :
        
    train_dataset = DroneDataset(

        data_path= cfg_data['data_path'],
        img_folder=  cfg_data['img_folder'],
        mask_folder= cfg_data['mask_folder'],

        type='TRAIN',

        transform=cfg_transform,
        augmentation=cfg_augmentation,
        data=cfg_data,
    )
    
    val_dataset = DroneDataset(

        data_path= cfg_data['data_path'],
        img_folder=  cfg_data['img_folder'],
        mask_folder= cfg_data['mask_folder'],

        type='VAL',

        transform=cfg_transform,
        augmentation=cfg_augmentation,
        data=cfg_data,
    )
    
    test_dataset = DroneDataset(

        data_path= cfg_data['data_path'],
        img_folder=  cfg_data['img_folder'],
        mask_folder= cfg_data['mask_folder'],

        type='TEST',

        transform=cfg_transform,
        augmentation=cfg_augmentation,
        data=cfg_data,
    )

    return train_dataset, val_dataset, test_dataset

def get_tools(cfg_train, cfg_distrib, model, rank=0) :
    device = None if cfg_distrib['active'] == 1 else torch.device(cfg_train['device'])
    print(f"Using device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['learning_rate'])
    criterion = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg_train['factor'], patience=cfg_train['patience']) if cfg_train['scheduler'] == 'ReduceLROnPlateau' else  None
    return device, optimizer, criterion, scheduler

def get_model(cfg_model) :
    if cfg_model['type'] == "MultiUnet" :
        model = Multi_Unet(num_classes=cfg_model['num_classes'], 
                       enc_name="resnet18",
                       enc_weights=cfg_model['enc_weights'],
                       ) 
    elif cfg_model['type'] == "SegFormer" :
        model = SegFormer(enc_name=cfg_model['enc_weights'], n_classes=cfg_model['num_classes'])
    elif cfg_model['type'] == "UFormer" :
        if cfg_model['enc_weights'] == 'best' :
            model = UFormer(trained=True, path="../gricad/")
        else :    
            model = UFormer(trained=False)
    elif cfg_model['type'] == "UnetBinary":
        model = UnetBinary(nb_classes=cfg_model["num_classes"])
    elif cfg_model['type'] == "BinarySegFormer":
        model = BinarySegFormer(enc_name=cfg_model['enc_weights'], n_classes=cfg_model['num_classes'])
    elif cfg_model['type'] == "BinaryUFormer" :
        if cfg_model['enc_weights'] == 'best' :
            model = BinaryUFormer(trained=True, path="")
        else :    
            model = BinaryUFormer(trained=False)         
    return model

def main_single_gpu(config):

    # Import configs :
    cfg_train = config['training']
    cfg_model = config['model']
    cfg_data = config['data']
    cfg_augmentation = config['augmentation']
    cfg_transform = config['transform']
    cfg_torch = config['torch']
    cfg_distrib = config['distributed']

    # Init the torch env :
    init_torch_env(cfg_torch)
    
    # Data Loading :
    train_dataset, val_dataset, test_dataset = get_dataset(cfg_data, cfg_transform, cfg_augmentation)
    print(f"TRAIN : {train_dataset.__len__()}")
    print(f"VAL : {val_dataset.__len__()}")
    print(f"TEST : {test_dataset.__len__()}")
    
    # Model :
    model = get_model(cfg_model)
    
    # Tools :
    device, optimizer, criterion, scheduler = get_tools(cfg_train, cfg_distrib, model)
    model.to(device).freeze_encoder()
    
    # Trainer
    model_tr = Trainer(model, config, train_dataset, val_dataset, test_dataset, criterion, optimizer, scheduler, device)

    # Execution :    
    train_hist = model_tr.fit(cfg_train['epochs'])
    test_hist = model_tr.test()
    
    # Historics :
    train_hist.to_csv(cfg_train['output_path'] + cfg_model['type'] + '/' + cfg_train['train_hist_name'] )
    test_hist.to_csv(cfg_train['output_path'] + cfg_model['type'] + '/' + cfg_train['test_hist_name'])
    
    return 0
 
def init_ddp(cfg_distrib) : 
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    init_process_group(backend=cfg_distrib['backend'])
    return rank

    
def main_multi_gpu(config):

    # Import configs :
    cfg_train = config['training']
    cfg_model = config['model']
    cfg_data = config['data']
    cfg_transform = config['transform']
    cfg_torch = config['torch']
    cfg_distrib = config['distributed']
    cfg_augmentation = config['augmentation']
    
    # init :
    rank = init_ddp(cfg_distrib)
    
    # Init the torch env :
    init_torch_env(cfg_torch)
    
    # Data Loading :
    train_dataset, val_dataset, test_dataset = get_dataset(cfg_data, cfg_transform, cfg_augmentation)
    print(f"TRAIN : {train_dataset.__len__()}")
    print(f"VAL : {val_dataset.__len__()}")
    print(f"TEST : {test_dataset.__len__()}")
    
    # Model :
    model = get_model(cfg_model)
    
    # Tools :
    _, optimizer, criterion, scheduler = get_tools(cfg_train, cfg_distrib, model)
    model.freeze_encoder()
    
    # Trainer
    model_tr = Trainer(model, config, train_dataset, val_dataset, test_dataset, criterion, optimizer, scheduler, _)

    # Execution : 
    train_hist = model_tr.fit(cfg_train['epochs'])
    
    # Historics :
    if rank == 0 :
        test_hist = model_tr.test()
        train_hist.to_csv(cfg_train['output_path'] + cfg_model['type'] + '/' + cfg_train['train_hist_name'] )
        test_hist.to_csv(cfg_train['output_path'] + cfg_model['type'] + '/' + cfg_train['test_hist_name'])
        
    destroy_process_group()
    
    return 0
       
    
if __name__ == "__main__":
    config_path="config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['distributed']['active'] == 0 :
        if 0 == main_single_gpu(config) :
            print(f"Training Ended Successfully !")
            
    elif config['distributed']['active'] == 1 :
        main_multi_gpu(config)
        print(f"Training Ended Successfully !")
        
