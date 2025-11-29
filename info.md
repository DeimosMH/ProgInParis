# Info about 

Rating evaluation:

1. 
2.
3. 
4.  


- video - yt/loom
- presentation - max 4 min


beta - blocking, concentration
gamma - sync of the microcircuits

- signal are noise - think about the filtering 

## Create venv


### Arch
```sh
sudo pacman -Syu
sudo pacman -S python python-pip
python -m venv ./venv
source ./venv/bin/activate
 # Should show Python 3.13.x

yay -S python313
python3.13 -m venv ./venv
source ./venv/bin/activate

yay -S pyenv
pyenv install 3.13.0
pyenv local 3.13.0
python -m venv ./venv
source ./venv/bin/activate

sudo pacman -S python-pyqt5
```

### Debian
```sh
# Method 1: Using deadsnakes PPA (Recommended)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv

# Create and activate venv
python3.13 -m venv ./venv
source ./venv/bin/activate

# Method 2: Build from source (if PPA unavailable)
sudo apt update
sudo apt install build-essential zlib1g-dev libffi-dev libssl-dev
wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz
tar -xzf Python-3.13.0.tgz && cd Python-3.13.0
./configure --enable-optimizations
make -j$(nproc) && sudo make altinstall
cd .. && python3.13 -m venv ./venv && source ./venv/bin/activate
```

### Apple
```sh
# Method 1: Using Homebrew
brew update
brew install python@3.13

# Create and activate venv
python3.13 -m venv ./venv
source ./venv/bin/activate

# Method 2: Using pyenv (for version management)
brew install pyenv
pyenv install 3.13.0
pyenv local 3.13.0
python -m venv ./venv
source ./venv/bin/activate
```

## Dependencies