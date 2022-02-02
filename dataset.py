import numpy as np
import pandas as pd
import cv2
from glob import glob
import json 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', validation_ratio = 0.1, transform = None):
        self.label_description = {
            "1_00_0" : "딸기", 
            "2_00_0" : "토마토",
            "2_a5_2" : "토마토_흰가루병_중기",
            "3_00_0" : "파프리카",
            "3_a9_1" : "파프리카_흰가루병_초기",
            "3_a9_2" : "파프리카_흰가루병_중기",
            "3_a9_3" : "파프리카_흰가루병_말기",
            "3_b3_1" : "파프리카_칼슘결핍_초기",
            "3_b6_1" : "파프리카_다량원소결필(N)_초기",
            "3_b7_1" : "파프리카_다량원소결필(P)_초기",
            "3_b8_1" : "파프리카_다량원소결필(K)_초기",
            "4_00_0" : "오이",
            "5_00_0" : "고추",
            "5_a7_2" : "고추_탄저병_중기",
            "5_b6_1" : "고추_다량원소결필(N)_초기",
            "5_b7_1" : "고추_다량원소결필(P)_초기",
            "5_b8_1" : "고추_다량원소결필(K)_초기",
            "6_00_0" : "시설포도",
            "6_a11_1" : "시설포도_탄저병_초기",
            "6_a11_2" : "시설포도_탄저병_중기",
            "6_a12_1" : "시설포도_노균병_초기",
            "6_a12_2" : "시설포도_노균병_중기",
            "6_b4_1" : "시설포도_일소피해_초기",
            "6_b4_3" : "시설포도_일소피해_말기",
            "6_b5_1" : "시설포도_축과병_초기"}
        
        self.csv_features = [
            '내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', 
            '내부 습도 1 평균', '내부 습도 1 최고', '내부 습도 1 최저',
            '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
        
        self.csv_feature_dict = {
            '내부 온도 1 평균': [3.4, 47.3],
            '내부 온도 1 최고': [3.4, 47.6],
            '내부 온도 1 최저': [3.3, 47.0],
            '내부 습도 1 평균': [0.0, 100.0],
            '내부 습도 1 최고': [0.0, 100.0],
            '내부 습도 1 최저': [0.0, 100.0],
            '내부 이슬점 평균': [0.1, 34.5],
            '내부 이슬점 최고': [0.2, 34.7],
            '내부 이슬점 최저': [0.0, 34.4]}

        self.label_encoder = {key:idx for idx, key in enumerate(self.label_description)}
        self.label_decoder = {val:key for key, val in self.label_encoder.items()}
        self.transform = transform
        self.mode = mode

        if mode == "train":
            self.files_dir, _ = train_test_split(sorted(glob(data_dir + "/train/*")), test_size = validation_ratio, random_state = 517)
        elif mode == "validation":
            _, self.files_dir = train_test_split(sorted(glob(data_dir + "/train/*")), test_size = validation_ratio, random_state = 517)
        elif mode == "test":
            self.files_dir = sorted(glob(data_dir + "/test/*"))


    def __len__(self):
        return len(self.files_dir)
    
    def __getitem__(self, i):
        file = self.files_dir[i]
        file_name = file.split('/')[-1]
        
        json_path = f'{file}/{file_name}.json'
        image_path = f'{file}/{file_name}.jpg'
        csv_path = f'{file}/{file_name}.csv'
        
        df = pd.read_csv(csv_path)[self.csv_features]
        df = df.replace('-', np.NaN)
        len_df = len(df)

        # MinMax scaling
        for col in self.csv_feature_dict.keys():
            df[col] = df[col].astype('float')
            df[col] = (df[col] - self.csv_feature_dict[col][0]) / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
        
        df = df.interpolate(method = "linear")
        df = df.iloc[::-1]

        if len_df < 590:
            df = pd.concat([pd.DataFrame(np.zeros((590 - len_df, 9)), columns = self.csv_features), df], axis = 0).reset_index(drop = 'index')
        else:
            df = df.iloc[-590:].reset_index(drop = 'index')
        df = df.fillna(0)
        
           
        img = cv2.imread(image_path)
        
        if self.mode == 'train' or self.mode == "validation":
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file["annotations"]["crop"]
            disease = json_file["annotations"]["disease"]
            risk = json_file["annotations"]["risk"]
            label = f'{crop}_{disease}_{risk}'
            
            if self.transform:
                transformed = self.transform(image = img)
                img = transformed["image"]

            return {
                'img' : img,
                'csv' : torch.tensor(df.to_numpy(), dtype=torch.float32),
                'label' : torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img' : img,
                'csv' : torch.tensor(self.csv_features, dtype=torch.float32)
            }
