# Continuos-Time Generative Models Workshop (CTGMWorkshop)

This is repository of developmnet and training code of various continuos-time generative models like diffusion and flow matching models.

[Work in progress]


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
