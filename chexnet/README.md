To setup environment:

```
python -m venv .venv
.venv\Scripts\activate
pip install poetry
cd .\chexnet\ 
poetry install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Run `python .\main.py` to check if it is working. Took ~8 mins to run on my 3060

Download the dataset from here: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

Project Directory:
```
├── .venv
├── chexnet
|   ├── Main.py
|   ├── ChexnetTrainer.py
|   ├── DatasetGenerator.py
|   ├── DensenetModels.py
|   ├── Main.py
|   ├── tar files
├── raw_data
|   ├── archive
|   |   ├── images_001
|   |   ├── etc...
├── poetry.lock
├── pyproject.toml
```