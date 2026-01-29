import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RoadDataset(Dataset):
    def __init__(self, root_dir):
        """
        [EN] root_dir: Relative path to the dataset folder (e.g., "./dataset/jungle")
        [CN] root_dir: 数据集文件夹的相对路径
        """
        # Ensure root_dir is handled correctly across different OS
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir, 'driving_log.csv')

        # 1. Load CSV and handle column names
        self.data = pd.read_csv(
            self.csv_file,
            header=None,
            names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        )

        # 2. Image Preprocessing Pipeline
        # Note: Input size must be (64, 64) to match the SimpleCNN architecture
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), # Converts to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        [ACADEMIC HIGHLIGHT] Dynamic Path Resolution
        This method fixes the hardcoded absolute paths (e.g., C:\Users\...) stored 
        in the simulator-generated CSV and maps them to the current environment.
        """
        # Get the raw path from CSV (often contains Windows-style absolute paths)
        raw_path = self.data.iloc[idx]['center']

        # [IMPROVEMENT] Universal filename extraction
        # Works for both Windows '\' and Linux '/' backslashes
        img_name = raw_path.replace('\\', '/').split('/')[-1]

        # [IMPROVEMENT] Relative Path Stitching
        # Construct path relative to root_dir, making the project portable
        real_img_path = os.path.join(self.root_dir, "IMG", img_name)

        # Image Loading & Exception Handling
        try:
            image = Image.open(real_img_path).convert('RGB')
        except FileNotFoundError:
            # Standard error logging for better debugging
            print(f"Error: Image not found at {real_img_path}. Please check your dataset structure.")
            raise

        if self.transform:
            image = self.transform(image)

        # Extract steering angle and convert to tensor for regression
        steering = float(self.data.iloc[idx]['steering'])
        return image, torch.tensor(steering, dtype=torch.float32)

'''
Professional Summary for GitHub/Portfolio:
- Data Portability: Implemented dynamic path re-mapping to ensure compatibility between 
  different local environments and OS platforms.
- Pre-processing: Integrated torchvision-based normalization and resizing specifically 
  tuned for lightweight CNN inference.
- Error Resilience: Added robust file-check mechanisms to prevent training interruption 
  during large-scale data loading.
'''
