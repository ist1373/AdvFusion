import transformers.adapters.composition as ac
from transformers import AdapterConfig
import logging
import os
import numpy as np
import torch.nn as nn
from transformers.adapters.composition import Fuse
from transformers.adapters import LoRAConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class AdapterUtils:

    def __init__(self) -> None:
        self.adapter_config = "pfeiffer"
    


    def add_new_adapter(self, model ,adapter_name, adapter_config = "pfeiffer"):
        if adapter_config == "pfeiffer":
            adapter_config = AdapterConfig.load(
                self.adapter_config)
            
        elif adapter_config == "lora":
            adapter_config = LoRAConfig(r=8, alpha=16)

        model.add_adapter(adapter_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        # Freeze all model weights except of those of this adapter
        model.train_adapter([adapter_name])
        logger.info(f"Adapter {adapter_name} is added.")
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(adapter_name)
    
    def add_adapter_fusion(self, model, adapters_path:list,with_trainable_weights = True):
        adapters_list = []
        for each_adapter_path in adapters_path:
            adapter_name = model.load_adapter(each_adapter_path,load_as = each_adapter_path.split("/")[-1], with_head = False)
            adapters_list.append(adapter_name)
        adapter_setup = Fuse(*adapters_list)
        model.add_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        if with_trainable_weights:
            model.train_adapter_fusion(adapter_setup)
        return adapters_list
    
    def zero_init_adapter_weights(self, model, adapter_name,device):
    #    print(model)
       for i in range(12):
          adapter = getattr(model.encoder.layer[i].output.adapters,adapter_name)
          adapter.adapter_up = nn.Linear(48, 768).to(device)
          adapter.adapter_up.weight.data.fill_(0)
          adapter.adapter_up.bias.data.fill_(0)
          adapter.adapter_up.weight.requires_grad = False
          adapter.adapter_up.bias.requires_grad = False
          adapter.adapter_down[0] = nn.Linear(768,48).to(device)
          adapter.adapter_down[0].weight.data.fill_(0)
          adapter.adapter_down[0].bias.data.fill_(0)


    def load_adapter_fusion(self,model,adapters_path,fusion_adapter_path,with_trainable_weights = False):
        adapters_list = []
        for each_adapter_path in adapters_path:
            adapter_name = model.load_adapter(each_adapter_path,load_as = each_adapter_path.split("/")[-1], with_head = False)
            adapters_list.append(adapter_name)
        adapter_setup = Fuse(*adapters_list)
        model.set_active_adapters(adapter_setup)
        # if with_trainable_weights:
        model.train_adapter_fusion(adapter_setup)
        model.load_adapter_fusion(fusion_adapter_path)
        return adapters_list
    
    def load_adapter(self, model, adapter_path, with_trainable_weights = False, with_head = False):
        adapter_name = model.load_adapter(adapter_path,with_head)
        if with_trainable_weights:
            model.set_active_adapters(adapter_name)
        logger.info("adapter is loaded successfully.")
        return adapter_name

    def print_trainable_parameters(self, model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("  total number of trainable params = %d", params)

    def save_adapter(self,model,adapter_name,output_dir):
        output_path = os.path.join(output_dir, adapter_name)                      
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        logger.info(f" {adapter_name} is saved successfully in {output_path}")
        model.save_adapter(output_path,adapter_name) 
    
    def save_adapter_fusion(self,model,output_dir,adapters_list):
        output_path = os.path.join(output_dir, "fusion")                      
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        logger.info(f"fusion adapter is saved successfully in {output_path}")
        model.save_adapter_fusion(output_path,adapters_list) 

    
