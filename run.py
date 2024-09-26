import modal
import os
import sys
import shlex

# Define the app
app = modal.App("stable-diffusion-webui")

# Handle volume creation
try:
    volume = modal.NetworkFileSystem.from_name("stable-diffusion-webui")
    print("Volume found.")
except modal.VolumeNotFound:
    volume = modal.NetworkFileSystem.new().persisted("stable-diffusion-webui")
    print("Volume created.")

@app.function(
    image=modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt update -y && "
        "apt install -y software-properties-common && "
        "apt update -y && "
        "add-apt-repository -y ppa:git-core/ppa && "
        "apt update -y && "
        "apt install -y git git-lfs && "
        "git --version && "
        "apt install -y aria2 libgl1 libglib2.0-0 wget && "
        "pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && "
        "pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    ),
    network_file_systems={"/content/stable-diffusion-webui": volume},
    gpu="T4",
    timeout=60000,
)
async def run():
    print("Starting run...")
    os.system("git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui")
    os.chdir("/content/stable-diffusion-webui")
    os.system("git reset --hard")

    # Download models using aria2
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/counterfeit-xl/resolve/main/counterfeitxl_v10.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o counterfeitxl_v10.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/juggernaut-xl/resolve/main/juggernautXL_version2.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o juggernautXL_version2.safetensors")
    os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sd_xl_refiner_1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o sd_xl_refiner_1.0_0.9vae.safetensors")

    os.environ['HF_HOME'] = '/content/stable-diffusion-webui/cache/huggingface'

    # Prepare for launching the application
    sys.path.append('/content/stable-diffusion-webui')
    sys.argv = shlex.split("--cors-allow-origins=* --xformers --theme dark --gradio-debug --share")

    from modules import launch_utils
    launch_utils.startup_timer.record("initial startup")
    launch_utils.prepare_environment()
    launch_utils.start()

@app.local_entrypoint()
def main():
    run.remote()
