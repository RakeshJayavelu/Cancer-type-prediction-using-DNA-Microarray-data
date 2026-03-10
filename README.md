# Cancer Classifier MLOps Project

Leukemia (ALL vs AML) classification using gene expression microarray data.
Converted from `task1.ipynb` into a structured MLOps project.

## Data

Place your raw CSV files in `data/raw/`:
- `actual.csv`
- `data_set_ALL_AML_train.csv`
- `data_set_ALL_AML_independent.csv`

## Project Structure

```
energy-mlops/
├── data/
│   ├── raw/            
│   └── processed/     
├── src/
│   ├── ingest.py       
│   ├── preprocess.py   
│   ├── train.py        
│   ├── evaluate.py     
│   └── predict.py      
├── api/
│   └── main.py         
├── run.py              
└── requirements.txt
```

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run full pipeline
```bash
python run.py
```

### Start API server
```bash
uvicorn api.main:app --reload
```

### API endpoint
```
POST /predict
{
  "features": [val1, val2, ...],   # 7129 raw gene expression values
  "model_name": "gaussian"         # "gaussian" | "svm" | "rf"
}
```
