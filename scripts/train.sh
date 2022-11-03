mkdir -p models/

# JAAD_all
python3 run.py --config_file configs_all.yaml
python3 run.py --config_file configs_all.yaml --resume --data_augmentation
python3 run.py --config_file configs_all.yaml --resume --auxiliary_loss

# JAAD_beh
python3 run.py --config_file configs_beh.yaml
python3 run.py --config_file configs_beh.yaml --resume --data_augmentation
python3 run.py --config_file configs_beh.yaml --resume --auxiliary_loss

# PIE
python3 run.py --config_file configs_pie.yaml
python3 run.py --config_file configs_pie.yaml --resume --auxiliary_loss
