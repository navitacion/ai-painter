import os
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
import torch

from src.models.cycle_gan import CycleGAN_Unet_Generator
from src.utils.utils import get_binary_file_downloader_html
from src.utils.transforms import ImageTransform

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
    st.markdown("v1: You can use the painter's style written below")
    st.markdown("- Monet\n- Van Gogh")
    st.markdown('---')

    style = st.sidebar.selectbox('Select Style', ('monet', 'vangogh'))

    net = model_builder(style)
    uploaded_file = st.sidebar.file_uploader("Choose Your Image")

    transform = ImageTransform()

    if uploaded_file is None:
        st.info("Please upload your image from sidebar list.\n\n After uploading, you can get the painter's style image soon.")
    else:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        w, h = image.size

        st.subheader('Input Image')
        st.image(image, use_column_width=True)

        inp = transform(image, phase='test')

        out = net(inp.unsqueeze(0))

        # Reverse Normalization
        out = out * 0.5 + 0.5
        out = out * 255

        out = out.squeeze()
        out = out.permute(1, 2, 0).detach().numpy().astype(np.uint8)

        out = Image.fromarray(out)
        out.save('output.jpg')
        out = out.resize((w, h))

        st.markdown('---')
        st.subheader('Output Image')
        st.image(out, use_column_width=True)

        tmp_download_link = get_binary_file_downloader_html('output.jpg', 'Picture')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    app()
