

## Api key
https://app.aihub.qualcomm.com/account/

```bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda create python=3.10 -n venv

conda init --all

conda activate venv

pip3 install qai-hub
pip3 install "qai-hub[torch]"

qai-hub configure --api_token <INSERT_API_TOKEN>
```

