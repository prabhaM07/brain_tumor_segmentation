import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Define the DoubleConvBlock and UNetModel classes (include the full definitions here)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, 
            padding=1, bias=False
        )

        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        # First Convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        # Second Convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        
        return self.relu2(x)

class UNetModel(nn.Module):
    def __init__(
        self, in_channels:int=3, 
        out_channels:int=1, 
        block_sizes:Tuple[int]=(64, 128, 256, 512)
    ):
        super(UNetModel, self).__init__()
        # Initialise model encoder & decoder using torch ModuleLists
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Create Encoder
        for block_size in block_sizes:
            self.encoder.append(DoubleConvBlock(in_channels, block_size))
            in_channels = block_size
            
        # Create Decoder
        for block_size in block_sizes[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(2 * block_size, block_size, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConvBlock(2 * block_size, block_size))
        
        # Create Bottleneck
        last_blocksize = block_sizes[-1]
        self.bottleneck = DoubleConvBlock(last_blocksize, 2 * last_blocksize)
  # Create Output Layer
        self.output_conv = nn.Conv2d(block_sizes[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        concatenations = []
        concatenations.append(x)
        
        # Propagate input downstream (Encode Input)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            concatenations.append(x)
            x = self.max_pool(x)
            
        # Execute bottleneck
        x = self.bottleneck(x)
        concatenations = concatenations[::-1]
        
        # Propagate input upstream (Decode Input) & concatenate layers
        for _ in range(0, len(self.decoder), 2):
            x = self.decoder[_](x)
            encoder_layer = concatenations[_ // 2]
            
            # Concatenate corrensponding encoder layer to decoder layer output
            concat_layer = torch.cat(
                (encoder_layer, x), dim=1
            )
            
            x = self.decoder[_ + 1](concat_layer)
            
        # Return predicted logits    
        return self.output_conv(x)

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Trained_Model.pth'  # Adjust this path as necessary
unet_model = UNetModel()
state_dict = torch.load(model_path, map_location=device)
state_dict = remove_module_prefix(state_dict)
unet_model.load_state_dict(state_dict)
unet_model.to(device)
unet_model.eval()
def predict_mask(model, image_path, device, color_map='rgb', title_size=16):
    # Read the image using PIL
    image = Image.open(image_path)
    
    # Convert to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    
    # Predict using the model
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = torch.sigmoid(pred_mask).cpu()
        pred_mask = (pred_mask > 0.5).float()  # Threshold of 0.5
    
    # Create columns for input and output images
    col1, col2, col3 = st.columns([1, 2, 1])  # Adding extra columns for centering
   
    with col2:
        st.header("Generated Mask", anchor=None)
        gen_mask = np.dstack([pred_mask[0][0]*0.1, pred_mask[0][0]*0.45, pred_mask[0][0]*0.1])
        fig, ax = plt.subplots(figsize=(14, 20))
        if color_map == 'rgb':
            ax.imshow(image + gen_mask)
        else:
            ax.imshow(image + gen_mask, cmap=color_map)
        
        st.pyplot(fig)

def app():
    # Streamlit interface
    st.title("MRI Segmentation with UNet")

    uploaded_file = st.file_uploader("Choose a TIFF file", type="tif")

    if uploaded_file is not None:
        with open("temp_image.tif", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict mask
        predict_mask(unet_model, "temp_image.tif", device)
