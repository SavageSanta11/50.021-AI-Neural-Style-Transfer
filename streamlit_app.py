import streamlit as st
from streamlit_card import card
from streamlit_image_select import image_select
from pathlib import Path
import json
import os
import PIL as pil
import nst_mosaic
# import nst_vangogh
# import nst_picasso
# import gan_1
# import gan_2
# import gan_3
# import gan_4
# import gan_5

# UI configurations
st.set_page_config(page_title="Neural Style Transfer",
                   page_icon=":bridge_at_night:",
                   layout="wide")

# Placeholder for gallery
home = st.empty()
gen_image = st.empty()
gallery= st.empty()

def stylize_image(model_type, inference_config):
    stylization_functions = {
        'Mosaic': nst_mosaic.stylize_static_image,
        'Starry Night': nst_mosaic.stylize_static_image,
        'Wave Crop': nst_mosaic.stylize_static_image,
        'Giger Crop': nst_mosaic.stylize_static_image,
        # 'GAN-1': gan_1.stylize_static_image,
        # 'GAN-2': gan_2.stylize_static_image,
        # 'GAN-3': gan_3.stylize_static_image,
        # 'GAN-4': gan_4.stylize_static_image,
        # 'GAN-5': gan_5.stylize_static_image,  
    }
    
    # Check if the model type is valid
    if model_type in stylization_functions:
        # Customize inference_config based on the model type

        # TODO: Add configurations for other models as needed

        # Call the corresponding stylization function with updated inference_config
        return stylization_functions[model_type](inference_config)
    else:
        raise ValueError("Invalid model type.")

def show_home(submitted, output_image_path):
    gen_image.empty()
    gallery.empty()
    submitted = False
    
    with home.container():
        st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/023/900/029/non_2x/abstract-colorful-watercolor-background-watercolor-texture-digital-art-painting-illustration-hand-painted-watercolor-abstract-morning-light-wallpaper-it-is-a-hand-drawn-vector.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
        st.markdown("# :rainbow[Automating Visual Artistry]")
        
        with st.form("my_form"):
            
            st.info("**Hello! Bring out your inner Picasso here ↓**", icon="👋🏾")
            with st.expander("**Choose your model**"):
                style = st.selectbox('Style', ('Mosaic', 'Starry Night', 'Wave Crop', 'Giger Crop', 'GAN-1', 'GAN-2', 'GAN-3', 'GAN-4', 'GAN-5'))
                title = st.text_input("Write a title for your creation", value="Untitled")
                caption = st.text_input("Write a unique caption", value="My first painting")
                
            uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
            save_path = None
            if submitted:
                print(submitted)
                # Save the uploaded file to a temporary directory
                raw = 'temp_workspace/streamlit-replicate-img-app/raw'
                # Ensure the folder exists, if not create it
                os.makedirs(raw, exist_ok=True)
                
                if uploaded_image is not None:
                    save_path = Path(raw, uploaded_image.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_image.getvalue())
                    print("File uploaded successfully")
                    st.success(f'File {uploaded_image.name} is successfully saved at {save_path}')
                    
                else:
                    st.error('Error in submitting, please try again')
                    
        if submitted:    
            home.empty() 
            # TODO: A FUNCTION CALL -> call to function that takes in uploaded image at save_path and stores output image at some other path
            show_output(save_path, style, title, caption, output_image_path)

def generate_description(image_name, style, title, caption):
    description_file_path = 'temp_workspace/gallery/description.json'
    
    # Load existing descriptions if the file exists
    descriptions = {}
    if os.path.exists(description_file_path):
        with open(description_file_path, "r") as desc_file:
            descriptions = json.load(desc_file)
    
    # Update the descriptions with the new image information
    descriptions[image_name] = {"model_name": style, "art_title": title, "description": caption}
    
    # Write the updated descriptions back to the file
    with open(description_file_path, "w") as desc_file:
        json.dump(descriptions, desc_file, indent=4)

def show_output(input_image_path, model_type, title, caption, output_image_path):
    gallery.empty()
    home.empty()
  
    with gen_image.container():
        st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://t3.ftcdn.net/jpg/06/27/85/70/360_F_627857047_VDETCsSfRkZuo5Fzdy3eHL1ZGdhsqYv9.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
        st.markdown("# :rainbow[Automating Visual Artistry]")
        
        # Define common inference parameters
        checkpoint = None
        checkpoint_name = None
        if model_type == "Mosaic":
            checkpoint = "SavageSanta25/johnson-mosaic"
            checkpoint_name = "mosaic.pth"
        elif model_type == "Starry Night":
            checkpoint = "SavageSanta25/johnson-starrynight"
            checkpoint_name = "vg_starry_night.pth"
        elif model_type == "Wave Crop":
            checkpoint = "SavageSanta25/johnson-wavecrop"
            checkpoint_name = "wave_crop.pth"
        elif model_type == "Giger Crop":
            checkpoint = "SavageSanta25/johnson-gigercrop"
            checkpoint_name = "giger_crop.pth"

        inference_config = dict()
        inference_config = {
            'output_images_path': output_image_path,
            'content': input_image_path,
            'img_width': 500,
            'checkpoint': checkpoint,
            'checkpoint_name': checkpoint_name,
            'redirected_output': None
        }
        
        output_image_path = stylize_image(model_type, inference_config)
        generate_description(output_image_path, model_type, title, caption)
        print(output_image_path)
        st.image(output_image_path, caption="Generated Image 🎈", use_column_width=True)

def show_gallery():
    gen_image.empty()
    home.empty()
    
    with gallery.container():
        st.markdown("# :rainbow[Gallery]")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://b.rgbimg.com/users/x/xy/xymonau/600/nLICqmW.jpg");
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display the "Hello! View your Creations Here" message
        st.info("**Hello! View your Creations Here ↓**", icon="👋🏾")

        # Display the "Edit Gallery Settings" expander
        with st.expander("**Edit Gallery Settings**", expanded=True):
            n = st.number_input("Grid Width", 1, 5, 2)

        st.markdown("""<div style="padding-bottom: 15px;"></div>""", unsafe_allow_html=True)

        # Read all descriptions from the description file
        description_file_path = 'temp_workspace/gallery/description.json'
        if os.path.exists(description_file_path):
            with open(description_file_path, "r") as desc_file:
                descriptions = json.loads(desc_file.read())
        else:
            descriptions = {}
        
        # Display all images stored in the local folder
        raw = 'temp_workspace/gallery'
        paths = [(file_path, file_path.stat().st_mtime) for file_path in Path(raw).iterdir()
                 if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        sorted_image_paths = sorted(paths, key=lambda x: x[1], reverse=True)
        image_paths = [str(file_path) for file_path, _ in sorted_image_paths]

        if image_paths:
            groups = []
            for i in range(0, len(image_paths), n):
                groups.append(image_paths[i:i+n])

            for group in groups:
                cols = st.columns(n)
                for i, image_path in enumerate(group):

                    image_name = Path(image_path).name

                    # Check if the image_name exists in the descriptions dictionary
                    if image_name in descriptions:
                        model_name = descriptions[image_name].get("model_name", "")
                        art_title = descriptions[image_name].get("art_title", "")
                        description = descriptions[image_name].get("description", "")
                        print(model_name)
                        print(art_title)
                        print(description)
                    else:
                        model_name = ""
                        art_title = ""
                        description = ""
                    
                    # Construct caption including model name and optional description
                    caption_parts = [f"Model: {model_name}"]
                    if description:
                        caption_parts.append(f"Caption: {description}")
                    
                    caption = " - ".join(caption_parts)

                    with cols[i]:
                        st.markdown(f'<div style="text-align: center; padding-bottom: 10px;"><b>{art_title}</b></div>', unsafe_allow_html=True)
                        st.image(image_path, use_column_width=True)
                        st.markdown(f'<div style="text-align: center; padding-bottom: 20px;">{caption}</div>', unsafe_allow_html=True)
                        

        else:
            # Display a message if no images are found in the gallery folder
            st.write("No images available in the gallery.")

def main():
    submitted = False

    output_image_path = 'temp_workspace/gallery'
    page = st.sidebar.radio("Navigation", ("Home", "Gallery"))

    if page == "Home":
        show_home(submitted, output_image_path)
    elif page == "Gallery":
        show_gallery()

if __name__ == "__main__":
    main()
