import streamlit as st
import pandas as pd
import numpy as np

from torchvision.transforms import ToPILImage
import torch
import sys
# import the py file for loading the dataset
if "..\\121-layer\\src" not in sys.path:
    sys.path.insert(0,r'..\121-layer\src')
print(sys.path)
from custom_densenet import *
from preprocessing import *

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
        
## import the dataloaders
train_dataset, val_dataset,train_loader, val_loader,test_dataset, test_loader= get_data_loaders(data_dir='../raw_data/archive/', label_file='../raw_data/archive/CXR8-selected/Data_Entry_2017_v2020.csv')


st.title('Predict Pneumothorax of Test Dataset Batch')

batch_iterator = iter(test_loader)
batch = next(batch_iterator)
images, labels = batch
# st.write(f"Batch {i + 1}")
images = images.to(device)

model = load_model("some path to be defined")

# Add a button to iterate through the dataloader
if st.button('Next Batch'):
    try:
        batch = next(batch_iterator)
    
        images, labels = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        pred_labels = (outputs > 5/10).long().T.squeeze(0)
        tp_count = fp_count = fn_count = 0
        for j, (image, pred_class, label) in enumerate(zip(images, pred_labels, labels)):
            st.write(f"Image {j + 1}")
            st.write(f"Predicted class: {pred_class.item()}")
            st.write(f"Actual class: {label.item()}")

            if (pred_class == label):
                tp_count += 1
            elif (pred_class == 0 and label == 1):
                fn_count += 1
            elif (pred_class == 1 and label == 0):
                fp_count += 1
            image_numpy = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))  # Convert tensor to numpy array and transpose dimensions
            st.image(image_numpy, clamp = True, caption=f"Original Image", use_column_width=True)
        st.write(f"true pos {tp_count}, false pos {fp_count}, false neg {fn_count}")
    except StopIteration:
        st.write("No more batches available.")



# # Send the batch of images through the model 
# with torch.no_grad():
#     outputs = model(images)

# # Process the model output (e.g., get predicted class)
# # For demonstration purposes, let's assume 'output' is a tensor of probabilities
# # You can replace this with your actual post-processing logic
# pred_labels = (outputs > 5/10).long().T.squeeze(0)

# # Display the results
# for j, (image, pred_class, labels) in enumerate(zip(images, pred_labels, labels)):
#     st.write(f"Image {j + 1}")
#     st.write(f"Predicted class: {pred_class.item()}")
#     st.write(f"Actual class: {labels.item()}")
#     # st.image(ToPILImage(image.to("cpu")), caption=f"Original Image", use_column_width=True)
#     image_numpy = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
#     image_element = st.image(image_numpy, clamp = True, caption=f"Original Image", use_column_width=True)


## have a button that generates next image and so on


st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

uploaded_img = st.file_uploader('File uploader')

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
