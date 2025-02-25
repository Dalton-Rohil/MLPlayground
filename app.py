import streamlit as st
from pipeline import Pipeline
from modules import SyntheticDataLoader, Normalizer, LogisticModel, AccuracyEvaluator

st.title("ML Playground")
st.write("Build and run your ML pipeline!")

if st.button("Run Pipeline"):
    pipeline = Pipeline(
        data_loader=SyntheticDataLoader(),
        preprocessor=Normalizer(),
        model=LogisticModel(),
        evaluator=AccuracyEvaluator()
    )
    result = pipeline.run()
    st.write(f"Accuracy: {result:.2f}")