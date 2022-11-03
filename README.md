# TAMFormer: Multi-Modal Transformer with Learned Attention Mask for Early Intent Prediction
This repository hosts the code related to the paper: Nada Osman, Guglielmo Camporese, and Lamberto Ballan, “TAMFormer: Multi-Modal Transformer with Learned Attention Mask for Early Intent Prediction”

```
@article{osman2022tamformer,
  title={TAMFormer: Multi-Modal Transformer with Learned Attention Mask for Early Intent Prediction},
  author={Osman, Nada and Camporese, Guglielmo and Ballan, Lamberto},
  journal={arXiv preprint arXiv:2210.14714},
  year={2022}
}
```

## Requirments
Refere to the procedure described in [Early Pedestrian Intent Prediction](https://github.com/NadaSOsman/EarlyPedestrianActionPrediction/)

## Training
1. Model, data and input types configurations can be modified in `config_file/configs_all.yaml`, `config_file/configs_beh.yaml`, or `config_file/configs_pie.yaml`
2. To train the default model run the following, replacing <dataset> with "all" for JAAD_all, "beh" for JAAD_beh, and "pie" for PIE.
  ```python3 run.py --configs_file config_file/configs_<dataset>.yaml```
3. To train with data augmentation:
    `python3 run.py --configs_file config_file/configs_<dataset>.yaml --data_augmentation`
4. To train with the auxiliary loss:
    `python3 run.py --configs_file config_file/configs_<dataset>.yaml --auxiliary_loss`
5. To resume training:
    `python3 run.py --configs_file config_file/configs_<dataset>.yaml --resume`
  
## Testing
    `python3 run.py --configs_file config_file/configs_<dataset>.yaml --test`
