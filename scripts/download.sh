# Aesthetic Quality (AQ) metric model weights
mkdir -p pretrained/aesthetic_model
wget -O pretrained/aesthetic_model/sa_0_4_vit_l_14_linear.pth https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true

# Image Quality (IQ) metric model weights
mkdir -p pretrained/image_quality
wget -O pretrained/image_quality/musiq_spaq_ckpt-358bb6af.pth https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/c753dca46d76714c9bb7849eadc2bd5854716320/musiq_spaq_ckpt-358bb6af.pth

# ImageReward (IR) metric model weights
python -c 'import ImageReward as RM; RM.load("ImageReward-v1.0", device="cpu", download_root="pretrained/ImageReward")'
