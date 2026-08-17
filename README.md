```py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version # 3.13

python -m pip install --upgrade pip
pip install -r requirements.txt

uvicorn src.api.main:app --reload