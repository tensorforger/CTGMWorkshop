# Continuos-Time Generative Models Workshop (CTGMWorkshop)

This is repository of development and training code of various continuos-time generative models like diffusion and flow matching models.

Treat the collection of notebooks as *workshop*, not as production training code. There might be bugs.

## Latest experimet

*03.06.2026*: Trained a base **117M** parameter diffusion model on the whole **COCO-2017** dataset.

Here are some generated samples and there prompt:

|                                    |                                               |                                                     |
|------------------------------------|-----------------------------------------------|-----------------------------------------------------|
|![1](assets/sample_apple.png)       |![2](assets/sample_man.png)                    |![3](assets/sample_stop_sign.png)                    |
|An apple on the table.              |There is a man walking on the path in the park.|A red stop sign next to trees.                       |
|![4](assets/sample_bathroom.png)    |![5](assets/sample_pizza.png)                  |![6](assets/sample_two_dogs.png)                     |
|A clean bathroom.                   |A plate with pizza on the table.               |There are two black dogs on the field of green grass.|

For more training and architecture details see `notebooks/train_models/train_larger_unet.ipynb`


# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/tensorforger/CTGMWorkshop
cd CTGMWorkshop
```

## 2. Install Dependencies

```bash
# Create environment
conda create -n ctgmworkshop python=3.12 pip -y
conda activate ctgmworkshop

# Install PyTorch with CUDA support (adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
```

## 3. Download Models

While main generative models are trained from scratch here, some pre-trained parts like VAE and text encoders are still used in some experiments.

```bash
cd CTGMWorkshop
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

## 4. Download Dataset

The main training dataset of Text-Image pairs would be COCO2017 here with **591k** pairs.

Download from [http://images.cocodataset.org/zips/train2017.zip](http://images.cocodataset.org/zips/train2017.zip)

Or use `notebooks/1.download-coco-dataset.ipynb`


#  Contributing

This is a research-oriented project under active development.

You can report issues if you can't fix them by your own: [https://github.com/tensorforger/CTGMWorkshop/issues](https://github.com/tensorforger/CTGMWorkshop/issues).

