import streamlit as st
import pandas as pd
import numpy as np

from torchvision.transforms import ToPILImage
import torch
import sys
# import the py file for loading the dataset
if "..\\121-layer\\src" not in sys.path:
    sys.path.insert(0,r'..\121-layer\src')
# print(sys.path)
from custom_densenet import *
from preprocessing import *

## get predictions for this batch
def process_data(model, batch, confidence):
    ## pass data through model
    images, labels = batch
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    pred_labels = (outputs > confidence).long().T.squeeze(0)

    return images, pred_labels, labels

## calculate the scores for this batch
def calculate_stats(pred_labels, labels):
    tp_count = fp_count = fn_count = tn_count = 0
    labels = labels.to(device)

    tp_count += sum(torch.logical_and(pred_labels, labels))
    fp_count += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), pred_labels))
    fn_count += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), labels))
    tn_count = len(labels) - tp_count - fp_count - fn_count
    return tp_count, fp_count, fn_count, tn_count
    
## display the batch and results on screen
def display_batch(images, pred_labels, labels):
    
    ## show images as grid
    for j, (image, pred_class, label) in enumerate(zip(images, pred_labels, labels)):
        
        ## display images
        if j % 4 == 0:
            cols = st.columns(4)
        image_numpy = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))  # Convert tensor to numpy array and transpose dimensions
        if pred_class.item() == label.item():
            caption = f":green[Pred: {pred_class.item()}, Actual: {label.item()}]"
        else: 
            caption = f":red[Pred: {pred_class.item()}, Actual: {label.item()}]"
        cols[j%4].image(image_numpy, clamp = True, use_column_width=True)
        cols[j%4].caption(caption)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## import model
@st.cache_resource
def load_model(model_path):
    print('load model')
    with torch.no_grad():
        pathModel = r"..\notebooks\dnet_models\m-custom_dnet_binary_by_img_count_lr_1e-4_long.pth.tar"

        model = dense_net(1, training=False)
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
        
## get the iterator
@st.cache_resource
def get_iterator():
    print('getting testloader...')
    train_dataset, val_dataset,train_loader, val_loader,test_dataset, test_loader= get_data_loaders(data_dir='../raw_data/archive/', label_file='../raw_data/archive/CXR8-selected/Data_Entry_2017_v2020.csv')
    batch_iterator = iter(test_loader)
    return batch_iterator

@st.cache_resource
def get_batch_data(_batch_iterator):
    return next(_batch_iterator)

st.title('Predict Pneumothorax of Test Dataset Batch')

## setup data from iterator and model
batch_iterator = get_iterator()

## to start the batch initially 
if "batch" not in st.session_state:
    batch = get_batch_data(batch_iterator)
    st.session_state["batch"] = batch

batch = st.session_state["batch"]
model = load_model("some path to be defined")

conf = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.1) 

# Add a button to iterate through the dataloader
if st.button('Next Batch'):
    try:
        batch = next(batch_iterator)
        st.session_state["batch"] = batch
        
    except StopIteration:
        st.write("No more batches available.")

## setup streamlit data
images, pred_labels, labels = process_data(model, batch, conf)
tp_count, fp_count, fn_count, tn_count = calculate_stats(pred_labels, labels)

precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) != 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

st.write(f"Precision: :{"green" if precision >= 0.5 else "red"}[{precision:.2f}], \
           Recall: :{"green" if recall > 0.5 else "red"}[{recall:.2f}], \
           F1 Score: :{"green" if f1_score > 0.5 else "red"}[{f1_score:.2f}]")
st.write(f":green[True Positives: {tp_count}, True Negatives: {tn_count}]")
st.write(f":red[False Positives: {fp_count}, False Negatives: {fn_count}]")
display_batch(images, pred_labels, labels)