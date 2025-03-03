import streamlit as st
from pipeline import Pipeline
from modules import SyntheticDataLoader, CSVDataLoader, Normalizer, LogisticModel, AccuracyEvaluator, DecisionTreeModel
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeModel (if not already in modules.py)
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
evaluator = AccuracyEvaluator()

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
    results = pipeline.run(data)

    # Display results
    st.write(f"Accuracy: {results['accuracy']:.2f}")

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(results["confusion_matrix"], cmap="Blues")
    for (i, j), val in np.ndenumerate(results["confusion_matrix"]):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)