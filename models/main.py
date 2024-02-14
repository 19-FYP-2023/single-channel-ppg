import yaml
from custom_dataloader import PpgBpDataset

with open("../configs/configs.yaml", 'r') as file:
    config = yaml.safe_load(file)

root_dir = config["root_dir"]
data_info_file = config["data_info_file"]
data_dir = config["data_dir"]
samples_per_subject = config["samples_per_subject"]

ppg_bp_dataset = PpgBpDataset(root_dir, data_info_file, data_dir, samples_per_subject)
