import torch
import yaml
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from python.model import *  
from python.droneDataset import *  
from python.trainer import *

def load_model(config) :
    cfg_model = config['model']
    cfg_training = config['training']
    if cfg_model['type'] == "MultiUnet" :
        model = Multi_Unet(num_classes=cfg_model['num_classes'], 
                        enc_weights=cfg_model['enc_weights'],
                        )
    elif cfg_model['type'] == "SegFormer" :
        model = SegFormer(enc_name=cfg_model['enc_weights'], n_classes=cfg_model['num_classes'])
    elif cfg_model['type'] == "UFormer" :
        if cfg_model['enc_weights'] == 'best' :
            model = UFormer(trained=True)
            return model
        else :    
            model = UFormer(trained=False)
    elif cfg_model['type'] == "UnetBinary":
        model = UnetBinary(nb_classes=cfg_model["num_classes"])
    elif cfg_model['type'] == "BinarySegFormer":
        model = BinarySegFormer(enc_name=cfg_model['enc_weights'], n_classes=cfg_model['num_classes'])
    elif cfg_model['type'] == "BinaryUFormer" :
        if cfg_model['enc_weights'] == 'best' :
            model = BinaryUFormer(trained=True)
            return model
        else :    
            model = BinaryUFormer(trained=False)
    

            
    model.load_state_dict(torch.load(cfg_training['output_path'] + cfg_model['type'] + '/' + cfg_model['type'] + ".pt", map_location=torch.device(cfg_training['device']), weights_only=True))
    
    return model

def get_test_data(config) :
    cfg_data = config['data']
    cfg_transform = config['transform']
    cfg_augmentation = config['augmentation']
    
    test_dataset = DroneDataset(

        data_path= cfg_data['data_path'],
        img_folder=  cfg_data['img_folder'],
        mask_folder= cfg_data['mask_folder'],

        type='TEST',

        transform=cfg_transform,
        augmentation=cfg_augmentation,
        data=cfg_data,
    )
    return test_dataset

def main(config_path="config.yaml"):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import configs :
    cfg_training = config['training']
    cfg_test = config['test']
    
    # Get the device :
    device = torch.device(cfg_training['device'])
    print(f"Using device: {device}")
    
    # Load the model :
    model = load_model(config).to(device)
    
    # Load the test dataset :
    test_dataset = get_test_data(config)
    
    # Trainer :
    model_tr = Trainer(model, config, None, None, test_dataset=test_dataset, criterion=None, device=device, optimizer=None, scheduler=None)
    
    # Execution :
    model_tr.predict(cfg_test["test_indexes"])   
    
if __name__ == "__main__":
    main()
