# MuseTalk
Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:

Build environment
We recommend Python 3.10 and CUDA 11.7. Set up your environment as follows:

conda create -n MuseTalk python==3.10
conda activate MuseTalk
Install PyTorch 2.0.1
Choose one of the following installation methods:

# Option 1: Using pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Option 2: Using conda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
Install Dependencies
Install the remaining required packages:

pip install -r requirements.txt
Install MMLab Packages
Install the MMLab ecosystem packages:

pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
Setup FFmpeg
Download the ffmpeg-static package

Configure FFmpeg based on your operating system:

For Linux:

export FFMPEG_PATH=/path/to/ffmpeg
# Example:
export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
For Windows: Add the ffmpeg-xxx\bin directory to your system's PATH environment variable. Verify the installation by running ffmpeg -version in the command prompt - it should display the ffmpeg version information.

Download weights
You can download weights in two ways:

Option 1: Using Download Scripts
We provide two scripts for automatic downloading:

For Linux:

sh ./download_weights.sh
For Windows:

# Run the script
download_weights.bat
Option 2: Manual Download
You can also download the weights manually from the following links:

Download our trained weights
Download the weights of other components:
sd-vae-ft-mse
whisper
dwpose
syncnet
face-parse-bisent
resnet18
Finally, these weights should be organized in models as follows:

./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── musetalkV15
│   └── musetalk.json
│   └── unet.pth
├── syncnet
│   └── latentsync_syncnet.pt
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
    
Quickstart
Inference
We provide inference scripts for both versions of MuseTalk:

Prerequisites
Before running inference, please ensure ffmpeg is installed and accessible:

# Check ffmpeg installation
ffmpeg -version
If ffmpeg is not found, please install it first:

Windows: Download from ffmpeg-static and add to PATH
Linux: sudo apt-get install ffmpeg
Normal Inference
Linux Environment
# MuseTalk 1.5 (Recommended)
sh inference.sh v1.5 normal

# MuseTalk 1.0
sh inference.sh v1.0 normal
Windows Environment
Please ensure that you set the ffmpeg_path to match the actual location of your FFmpeg installation.

# MuseTalk 1.5 (Recommended)
python -m scripts.inference --inference_config configs\inference\test.yaml --result_dir results\test --unet_model_path models\musetalkV15\unet.pth --unet_config models\musetalkV15\musetalk.json --version v15 --ffmpeg_path ffmpeg-master-latest-win64-gpl-shared\bin

# For MuseTalk 1.0, change:
# - models\musetalkV15 -> models\musetalk
# - unet.pth -> pytorch_model.bin
# - --version v15 -> --version v1
Real-time Inference
Linux Environment
# MuseTalk 1.5 (Recommended)
sh inference.sh v1.5 realtime

# MuseTalk 1.0
sh inference.sh v1.0 realtime
Windows Environment
# MuseTalk 1.5 (Recommended)
python -m scripts.realtime_inference --inference_config configs\inference\realtime.yaml --result_dir results\realtime --unet_model_path models\musetalkV15\unet.pth --unet_config models\musetalkV15\musetalk.json --version v15 --fps 25 --ffmpeg_path ffmpeg-master-latest-win64-gpl-shared\bin

# For MuseTalk 1.0, change:
# - models\musetalkV15 -> models\musetalk
# - unet.pth -> pytorch_model.bin
# - --version v15 -> --version v1
The configuration file configs/inference/test.yaml contains the inference settings, including:

video_path: Path to the input video, image file, or directory of images
audio_path: Path to the input audio file
Note: For optimal results, we recommend using input videos with 25fps, which is the same fps used during model training. If your video has a lower frame rate, you can use frame interpolation or convert it to 25fps using ffmpeg.

Important notes for real-time inference:

Set preparation to True when processing a new avatar
After preparation, the avatar will generate videos using audio clips from audio_clips
The generation process can achieve 30fps+ on an NVIDIA Tesla V100
Set preparation to False for generating more videos with the same avatar
For faster generation without saving images, you can use:

python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images
