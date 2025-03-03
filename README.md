# MLPlayground

A Streamlit web app to experiment with modular machine learning pipelines.

## Features
- Build custom ML pipelines with interchangeable components.
- Supports synthetic data or CSV uploads.
- Choose preprocessors (Normalizer) and models (Logistic Regression, Decision Tree).

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/MLPlayground.git
   cd MLPlayground

## Set up virtual environment:
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

## Install dependencies:
pip install -r requirements.txt

## Run the app:
streamlit run app.py


## Usage
-pen your browser at localhost:8501.
-Select a data source (Synthetic or CSV).
-Choose a preprocessor and model, then click "Run Pipeline" to see the accuracy.

## Contributing
-Fork the repo and submit a pull request.
-Add new modules in modules.py—see existing ones for examples.

## License
MIT