# MLB Pitch Type Predictor

[Click on this link for an interactive readme of the project](https://alvinthai.github.io/MLB_pitch_type_predictor/readme.html).

---
Installation

1. Create a new Python 2.7 virtual environment for the project
```
conda create -n mlb python=2.7
```
2. Activate virtual enviornment
```
conda activate mlb
```
3. Clone repo and install required dependencies
```
pip install -r requirements.txt
```
4. Install jupyter kernel for virtual environment and move kernel files
```
pip install jupyter
python -m ipykernel install --name mlb --display-name "Python 2 (mlb)"
mv /usr/local/share/jupyter/kernels/mlb ~/Library/Jupyter/kernels/
```
5. Use ```Python 2 (mlb)``` kernel in jupyter notebook when importing files from this repo
