git clone https://github.com/Vchitect/VBench.git ./VBench
git -C ./VBench checkout --detach 98c21aa533342ec43e319cb3fedf46b51399a502

# VBench and detectron2 build against the installed torch, so they are
# installed after the environment is created.
python -m pip install --no-build-isolation ./VBench

CUDA_HOME="$CONDA_PREFIX" python -m pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"


MODELS=https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/83290ee59b4e2c0a4d198857edbc79dd186b8d44

mkdir -p pretrained/vbench/umt_model
wget -O pretrained/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth $MODELS/l16_ptk710_ftk710_ftk400_f16_res224.pth

mkdir -p pretrained/vbench/grit_model
wget -O pretrained/vbench/grit_model/grit_b_densecap_objectdet.pth $MODELS/grit_b_densecap_objectdet.pth

mkdir -p pretrained/vbench/ViCLIP
wget -O pretrained/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth $MODELS/ViClip-InternVid-10M-FLT.pth

# Tag2Text (scene)
mkdir -p pretrained/vbench/caption_model
wget -O pretrained/vbench/caption_model/tag2text_swin_14m.pth https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/7c76cec1377de0d46df8091a032c4b667aea4459/tag2text_swin_14m.pth

# AMT (motion_smoothness)
mkdir -p pretrained/vbench/amt_model
wget -O pretrained/vbench/amt_model/amt-s.pth https://huggingface.co/lalala125/AMT/resolve/31d0420ef7788f6dc22aad3445148705007141bb/amt-s.pth

# ViCLIP tokenizer vocabulary: VBench does not download it
cp VBench/vbench2_beta_trustworthiness/third_party/ViCLIP/bpe_simple_vocab_16e6.txt.gz pretrained/vbench/ViCLIP/

# Aesthetic predictor (aesthetic_quality): VBench's own fallback download
# saves it under a wrong file name
mkdir -p pretrained/vbench/aesthetic_model/emb_reader
wget -O pretrained/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
