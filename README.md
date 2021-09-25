<div align="center">    
 
# Frame-level Video Scene Detection

</div>
 
## Description   
This repository is for academic research "Frame-level Video Scene Detection". It includes the models we proposed and data preprocessing script.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/guyleaf/FLVCD

# install dependencies  
cd FLVCD
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd FLVCD

# run module (example: mnist as your main contribution)   
NEPTUNE_API_TOKEN="your_api_key" python trainer.py --lr 0.001 --base_folder /nfs/features/BBC --dataset BBC --d_model 1024 --d_inner 2048 --n_heads 4 --t_steps 50 --dropout 0.2 --inter_dropout 0.2 --layer_config ll --memory_profile --shuffle --max_epochs 200 --gpus 1 --deterministic --tags 1fps RMSprop xavier_uniform_for_linear_without_relu kaiming_uniform_for_linear_with_relu  
```

### Citation   
1. https://github.com/andreamad8/Universal-Transformer-Pytorch
2. https://github.com/tensorflow/tensor2tensor
