work_dir = "/Users/shaanpakala/Desktop/NSF_REU_2024/Research/"

import torch
import sys
sys.path.append(f'{work_dir}/')

data_folder = f"{work_dir}classification_datasets/"
meta_data_folder = f"{work_dir}training_tensors/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")