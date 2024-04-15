import torch
import pandas as pd
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader,random_split

torch.manual_seed(42)

class PneumoniaDataset(Dataset):
    def __init__(self,data_dir, label_file, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file)
        #if label contains Pneumothorax = 1 else 0 
        self.labels_df['Pneumothorax'] = self.labels_df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x.split('|') else 0)

        # Separate the majority and minority classes
        df_majority = self.labels_df[self.labels_df['Pneumothorax'] == 0]
        df_minority = self.labels_df[self.labels_df['Pneumothorax'] == 1]

        # Undersample the majority class without replacement
        majority_size = len(df_minority) # same as the minority class size
        df_majority_undersampled = df_majority.sample(n=majority_size, random_state=42)

        # Combine the minority class with the undersampled majority class
        self.labels_df = pd.concat([df_minority, df_majority_undersampled]).reset_index(drop=True)

        # Shuffle the dataset
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '*', 'images', '*.png'))}
        self.labels_df['path'] = self.labels_df['Image Index'].map(self.image_paths.get)
        self.labels_df.dropna(subset=['path'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['path']
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx]['Pneumothorax']
        if self.transform:
            image = self.transform(image)
        return image, label
    def get_label_1_paths(self):
        label_1_paths = self.labels_df[self.labels_df['Pneumothorax'] == 1]['path'].tolist()
        return label_1_paths

class PneumoniaDataset_v2(Dataset):
    def __init__(self,data_dir, label_file, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file)
        #if label contains Pneumothorax = 1 else 0 
        self.labels_df['Pneumothorax'] = self.labels_df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x.split('|') else 0)

        # Separate the majority and minority classes
        df_majority = self.labels_df[self.labels_df['Pneumothorax'] == 0]
        df_minority = self.labels_df[self.labels_df['Pneumothorax'] == 1]

        # Undersample the majority class without replacement
        majority_size = len(df_minority) # same as the minority class size
        df_majority_undersampled = df_majority.sample(n=majority_size, random_state=42)

        # Combine the minority class with the undersampled majority class
        self.labels_df = pd.concat([df_minority, df_majority_undersampled]).reset_index(drop=True)

        ## NEW add normalization to the follow up
        self.labels_df['FollowUp_normalized'] = (self.labels_df['Follow-up #'] - self.labels_df['Follow-up #'].min()) / (self.labels_df['Follow-up #'].max() - self.labels_df['Follow-up #'].min())

        # Shuffle the dataset
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '*', 'images', '*.png'))}
        self.labels_df['path'] = self.labels_df['Image Index'].map(self.image_paths.get)
        self.labels_df.dropna(subset=['path'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['path']
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx]['Pneumothorax']
        ## ADDITIONAL NON PICTORIAL DATA
        follow_up = self.labels_df.iloc[idx]['FollowUp_normalized']
        if self.transform:
            image = self.transform(image)
        return image, label, follow_up
    def get_label_1_paths(self):
        label_1_paths = self.labels_df[self.labels_df['Pneumothorax'] == 1]['path'].tolist()
        return label_1_paths

class PneumoniaDataset_v3(Dataset):
    def __init__(self,data_dir, label_file, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file)
        #if label contains Pneumothorax = 1 else 0 
        self.labels_df['Pneumothorax'] = self.labels_df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x.split('|') else 0)

        # Separate the majority and minority classes
        df_majority = self.labels_df[self.labels_df['Pneumothorax'] == 0]
        df_minority = self.labels_df[self.labels_df['Pneumothorax'] == 1]

        # Undersample the majority class without replacement
        majority_size = len(df_minority) # same as the minority class size
        df_majority_undersampled = df_majority.sample(n=majority_size, random_state=42)

        # Combine the minority class with the undersampled majority class
        self.labels_df = pd.concat([df_minority, df_majority_undersampled]).reset_index(drop=True)

        ## NEW add normalization to the follow up
        self.labels_df['FollowUp_normalized'] = (self.labels_df['Follow-up #'] - self.labels_df['Follow-up #'].min()) / (self.labels_df['Follow-up #'].max() - self.labels_df['Follow-up #'].min())
        self.labels_df['Age_normalized'] = (self.labels_df['Patient Age'] - self.labels_df['Patient Age'].min()) / (self.labels_df['Patient Age'].max() - self.labels_df['Patient Age'].min())
        self.labels_df['Gender_numeric'] = self.labels_df['Patient Gender'].map({'M': 0, 'F': 1})
        
        # Shuffle the dataset
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '*', 'images', '*.png'))}
        self.labels_df['path'] = self.labels_df['Image Index'].map(self.image_paths.get)
        self.labels_df.dropna(subset=['path'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['path']
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx]['Pneumothorax']
        ## ADDITIONAL NON PICTORIAL DATA
        non_img_data = torch.tensor(self.labels_df.iloc[idx][['FollowUp_normalized','Age_normalized','Gender_numeric']])
        if self.transform:
            image = self.transform(image)
        return image, label, non_img_data
    def get_label_1_paths(self):
        label_1_paths = self.labels_df[self.labels_df['Pneumothorax'] == 1]['path'].tolist()
        return label_1_paths

class PneumoniaDataset_grey(Dataset):
    def __init__(self,data_dir, label_file, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file)
        #if label contains Pneumothorax = 1 else 0 
        self.labels_df['Pneumothorax'] = self.labels_df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x.split('|') else 0)

        # Separate the majority and minority classes
        df_majority = self.labels_df[self.labels_df['Pneumothorax'] == 0]
        df_minority = self.labels_df[self.labels_df['Pneumothorax'] == 1]

        # Undersample the majority class without replacement
        majority_size = len(df_minority) # same as the minority class size
        df_majority_undersampled = df_majority.sample(n=majority_size, random_state=42)

        # Combine the minority class with the undersampled majority class
        self.labels_df = pd.concat([df_minority, df_majority_undersampled]).reset_index(drop=True)

        # Shuffle the dataset
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '*', 'images', '*.png'))}
        self.labels_df['path'] = self.labels_df['Image Index'].map(self.image_paths.get)
        self.labels_df.dropna(subset=['path'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['path']
        image = Image.open(img_name)
 
        label = self.labels_df.iloc[idx]['Pneumothorax']
        if self.transform:
            image = self.transform(image)
        return image, label
    def get_label_1_paths(self):
        label_1_paths = self.labels_df[self.labels_df['Pneumothorax'] == 1]['path'].tolist()
        return label_1_paths

class NormalizeGrayscale:
    def __call__(self, img):
        # Convert PIL image to tensor
        img_tensor = transforms.ToTensor()(img)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[[0]]
        # Calculate mean and standard deviation of the image tensor
        mean_value = torch.mean(img_tensor)
        std_value = torch.std(img_tensor)
        
        # Normalize the image tensor
        normalized_img = (img_tensor - mean_value) / std_value
        
        return normalized_img

class PneumoniaDataset(Dataset):
    def __init__(self,data_dir, label_file, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file)
        #if label contains Pneumothorax = 1 else 0 
        self.labels_df['Pneumothorax'] = self.labels_df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x.split('|') else 0)

        # Separate the majority and minority classes
        df_majority = self.labels_df[self.labels_df['Pneumothorax'] == 0]
        df_minority = self.labels_df[self.labels_df['Pneumothorax'] == 1]

        # Undersample the majority class without replacement
        majority_size = len(df_minority) # same as the minority class size
        df_majority_undersampled = df_majority.sample(n=majority_size, random_state=42)

        # Combine the minority class with the undersampled majority class
        self.labels_df = pd.concat([df_minority, df_majority_undersampled]).reset_index(drop=True)

        # Shuffle the dataset
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '*', 'images', '*.png'))}
        self.labels_df['path'] = self.labels_df['Image Index'].map(self.image_paths.get)
        self.labels_df.dropna(subset=['path'], inplace=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['path']
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx]['Pneumothorax']
        if self.transform:
            image = self.transform(image)
        return image, label
    def get_label_1_paths(self):
        label_1_paths = self.labels_df[self.labels_df['Pneumothorax'] == 1]['path'].tolist()
        return label_1_paths

def get_transforms():  
#resize the image size to 224 x 224 pixels
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # NormalizeGrayscale()
    ])
    return transform

def get_data_loaders(data_dir, label_file,batch_size=16, val_split=0.1,  test_split=0.1):

    transform = get_transforms()
    dataset = PneumoniaDataset(data_dir=data_dir, label_file=label_file, transform=transform)
    
    # Calculate split sizes
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size
    
    # Split the dataset into training, validation, and test sets
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset,val_dataset,train_loader,val_loader,test_dataset,test_loader

train_dataset, val_dataset,train_loader, val_loader,test_dataset, test_loader= get_data_loaders(data_dir='../raw_data/archive/', label_file='../raw_data/archive/CXR8-selected/Data_Entry_2017_v2020.csv')

print(f"Training Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")