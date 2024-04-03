# Neural Style Transfer

This project has 3 different implementations of Neural Style Transfer, divided into Stage 1, Stage 2 and Stage 3. More details to come.

## Stage 1

### Setup

On Windows, you can run the following commands:

    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
    cd stage1
    mkdir data
    cd data
    mkdir content-images
    mkdir style-images

Keep your content images and style images in their respective directories. ** I know this is a very bad way of doing things. Once we decide on what data to commonly use for all stages we can have the data directory outside**

### Running stage 
To get started, got to ``nst.py`` and change the ``content`` and ``style`` arguments based on the images you want to use. 

    python nst.py

