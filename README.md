# Neural Style Transfer

This project has 3 different implementations of Neural Style Transfer, divided into Stage 1, Stage 2 and Stage 3. More details to come.

## Stage 1

### Setup

On Windows, you can run the following commands:

    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
    cd stage1

### Inference
To get started, go to ``nst.py`` and change the ``--content`` and ``--style`` arguments based on the images you want to use.  Look in the ``data`` directory for content and style images. Then run:

    python nst.py

## Stage 2
Make sure you have run the setup instructions from ``Stage 1``.

### Training

1. Download the MSCOCO dataset from here: http://images.cocodataset.org/zips/train2014.zip and place it in the ``data/mscoco`` directory.
2. Change the ``--style_img_name`` argument based on your choice from the ``data/style-images`` directory. 
3. Run the following:

	    python train.py


4. Look in the ``checkpoints/stage2`` directory for your checkpoint when training is complete.

### Inference
1. Pick a checkpoint to use from the ``checkpoints/stage2`` directory. Checkpoints are named based on the style image they have been trained on.
2. Change ``--content`` argument based on your choice of content image from the ``data`` directory.
3. Run the following:

	    python nst.py
	    
4. Look in the ``data/output-images/stage2`` directory when inference is complete.