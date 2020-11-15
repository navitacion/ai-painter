import os
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.models.cycle_gan import CycleGAN_Unet_Generator
from src.utils.utils import ImageTransform

@st.cache
def model_builder(style='monet'):
    net = CycleGAN_Unet_Generator()
    weight_path = './weights'
    weight = torch.load(os.path.join(weight_path, f'{style}.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(weight)
    net = net.eval()

    return net



def app():
    # Title
    st.title('AI Painter')
    st.markdown('---')
    st.markdown('This app is a demo of CycleGAN')
    st.markdown('---')

    style = st.selectbox('Select GAN', ('monet', 'vangogh'))

    net = model_builder(style)
    uploaded_file = st.file_uploader("Choose a file")

    transform = ImageTransform()

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        w, h = image.size
        st.image(image, use_column_width=True)

        inp = transform(image, phase='test')

        out = net(inp.unsqueeze(0))

        # Reverse Normalization
        out = out * 0.5 + 0.5
        out = out * 255

        out = out.squeeze()
        out = out.permute(1, 2, 0).detach().numpy().astype(np.uint8)

        out = Image.fromarray(out)
        out = out.resize((w, h))

        st.image(out, use_column_width=True)


if __name__ == '__main__':
    app()
