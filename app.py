import streamlit as st
from pipeline import Pipeline
from modules import SyntheticDataLoader, CSVDataLoader, Normalizer, LogisticModel, AccuracyEvaluator
from sklearn.tree import DecisionTreeClassifier

# Additional model for variety
class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, data):
        self.model.fit(data["X"], data["y"])

    def predict(self, data):
        return self.model.predict(data["X"])

st.title("ML Playground")
st.write("Build and run your ML pipeline!")

# Data source selection
data_source = st.selectbox("Choose data source", ["Synthetic", "Upload CSV"])
uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# Module selection
preprocessor_option = st.selectbox("Choose preprocessor", ["Normalizer"])
model_option = st.selectbox("Choose model", ["Logistic Regression", "Decision Tree"])

# Map selections to classes
preprocessors = {"Normalizer": Normalizer()}
models = {"Logistic Regression": LogisticModel(), "Decision Tree": DecisionTreeModel()}
evaluator = AccuracyEvaluator()  # Fixed for now

if st.button("Run Pipeline"):
    # Data loader
    data_loader = SyntheticDataLoader() if data_source == "Synthetic" else CSVDataLoader()

    # Pipeline
    pipeline = Pipeline(
        data_loader=data_loader,
        preprocessor=preprocessors[preprocessor_option],
        model=models[model_option],
        evaluator=evaluator
    )
    data = uploaded_file if data_source == "Upload CSV" else None
    result = pipeline.run(data)
    st.write(f"Accuracy: {result:.2f}")