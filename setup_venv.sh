sudo apt install python3-pip
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gdown https://drive.google.com/file/d/1Iu6COsO1xYVdbrD-v31-IkyZde6209sB/view\?usp\=drive_link --fuzzy -O downloaded_file.zip
unzip downloaded_file.zip