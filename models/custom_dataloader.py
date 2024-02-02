import os
import torch
import numpy as np 
import matlab.pyplot as plt 
from torch.utils.data import Dataset, DataLoader 
from openpyxl import load_workbook

class PpgBpDataset(Dataset):
    def __init__(self, root_dir, data_info_file, data_dir, samples_per_subject, transform=None):
        self.root_dir = root_dir
        self.data_info_file = data_info_file
        self.data_dir = data_dir
        self.transform = transform 
        self.workbook = load_workbook(f"{self.root_dir}/{self.data_info_file}")
        self.info_sheet = self.workbook["cardiovascular dataset"]
        self.samples_per_subject = samples_per_subject
        

        header_row = self.info_sheet[2]
        header_row = [cell.value for cell in header_row]


        num_col_idx = header_row.index("Num.")
        subject_col_idx = header_row.index("subject_ID")
        sysbp_col_idx = header_row.index("Systolic Blood Pressure(mmHg)")
        diapb_col_idx = header_row.index("Diastolic Blood Pressure(mmHg)")

    def __len__(self):
        return self.info_sheet.cell(self.info_sheet.max_row, 0).value * self.samples_per_subject

    def __getitem__(self, idx):
        dataset_idx = int(idx/self.samples_per_subject)
        sample_num = idx%self.samples_per_subject + 1

        row_idx = 3 + dataset_idx
        subject_id = self.info_sheet.cell(row_idx, subject_col_idx).value
        sysbp = self.info_sheet.cell(row_idx, sysbp_col_idx).value
        diabp = self.info_sheet.cell(row_idx, dia_bp_idx).value

        data_sample_path = f"{self.root_dir}/{self.data_dir}/{subject_id}_{sample_num}.txt"
        data_sample = np.load(data_sample_path)

        return data_sample, subject_id, sysbp, diabp


