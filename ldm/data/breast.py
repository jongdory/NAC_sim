import os
import cv2
import numpy as np
import pandas as pd
import random
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from ldm.modules.midas.api import load_midas_transform
from scipy.ndimage import center_of_mass, distance_transform_edt
from einops import rearrange, repeat

def get_ACRIN():
    data = []
    data_path = ""
    train_path = data_path + "/Training"
    test_path = data_path + "/Testing"
    xlsx_path = data_path +"/Full Collection Ancillary Patient Information file.xlsx"

    sublist = os.listdir(train_path) + os.listdir(test_path)[:30]

    df = pd.read_excel(xlsx_path)

    for subject in sublist:
        sub_data = df[df['TCIA PATIENT ID'] == subject]

        pcr = sub_data['pcr'].values[0]
        her2 = sub_data['hrher4g'].values[0]

        if pcr == 'NA': continue
        if her2 == 'NA': continue
        if her2 == "HR + / HER2 +":  hr =  "HR+"; her2 = "HER2+"
        elif her2 == "HR + / HER2 -":  hr =  "HR+"; her2 = "HER2-"
        elif her2 == "HR - / HER2 +":  hr =  "HR-"; her2 = "HER2+"
        elif her2 == "HR - / HER2 - (TN)": hr =  "HR-"; her2 = "HER2-"

        data_path = train_path if subject in os.listdir(train_path) else test_path
        if not os.path.exists(os.path.join(data_path,subject,"T0/dce.png")): continue
        if not os.path.exists(os.path.join(data_path,subject,"T1/dce.png")): continue
        if not os.path.exists(os.path.join(data_path,subject,"T2/dce.png")): continue

        data.append({
            "image": os.path.join(data_path,subject),
            "subject_id": subject,
            "pcr": pcr,
            "hr": hr,
            "her2": her2
        })

    return data


def get_TestACRIN():
    data = []
    data_path = ""
    test_path = data_path + "/Testing"
    xlsx_path = data_path +"/Full Collection Ancillary Patient Information file.xlsx"

    sublist = os.listdir(test_path)[30:]

    df = pd.read_excel(xlsx_path)

    for subject in sublist:
        sub_data = df[df['TCIA PATIENT ID'] == subject]

        pcr = sub_data['pcr'].values[0]
        her2 = sub_data['hrher4g'].values[0]

        if pcr == 'NA': continue
        if her2 == 'NA': continue
        if her2 == "HR + / HER2 +":  hr =  "HR+"; her2 = "HER2+"
        elif her2 == "HR + / HER2 -":  hr =  "HR+"; her2 = "HER2-"
        elif her2 == "HR - / HER2 +":  hr =  "HR-"; her2 = "HER2+"
        elif her2 == "HR - / HER2 - (TN)": hr =  "HR-"; her2 = "HER2-"

        if not os.path.exists(os.path.join(test_path,subject,"T0/dce.png")): continue
        if not os.path.exists(os.path.join(test_path,subject,"T1/dce.png")): continue
        if not os.path.exists(os.path.join(test_path,subject,"T2/dce.png")): continue

        data.append({
            "image": os.path.join(test_path,subject),
            "subject_id": subject,
            "pcr": pcr,
            "hr": hr,
            "her2": her2
        })

    return data

class MyDataset(Dataset):
    def __init__(self):
        self.data = get_ACRIN()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item['image']

        time_t = ['T0', 'T1', 'T2']
        tag_t = random.choice(time_t)
        time_t.remove(tag_t)
        source = path + "/T0/dce.png"
        target = path + f"/{tag_t}/dce.png"

        pcr = item['pcr']
        hr = item['hr']
        her2 = item['her2']

        source = cv2.imread(source)
        target = cv2.imread(target)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if source.shape[1] != 256:
            source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_CUBIC)

        if np.random.rand() < 0.6:
            f = random.choice([0,1])
            source = np.flip(source, axis=f)
            target = np.flip(target, axis=f)
        if np.random.rand() < 0.6:
            k = random.randint(1, 3)
            source = np.rot90(source, k=k)
            target = np.rot90(target, k=k)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = f"{tag_t},{pcr},{hr},{her2}, "
        d_prompt = ""
        
        return dict(jpg=target, txt=prompt, dtxt=d_prompt, source=source)
    

class MyTestset(Dataset):
    def __init__(self, time):
        self.data = get_TestACRIN()
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item['image']
        subject_id = item['subject_id']

        tag_t = self.time
        source = path + "/T0/dce.png"
        target = path + f"/{tag_t}/dce.png"

        pcr = item['pcr']
        pcr = 'pCR' if pcr == 'Non-pcr' else 'Non-pCR'
        hr = item['hr']
        her2 = item['her2']

        source = cv2.imread(source)
        target = cv2.imread(target)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if source.shape[1] != 256:
            source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = f"{tag_t},{pcr},{hr},{her2}, "
        d_prompt = ""
        
        return dict(jpg=target, txt=prompt, dtxt=d_prompt, source=source, subject_id=subject_id)