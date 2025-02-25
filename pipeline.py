class Pipeline:
    def __init__(self, data_loader, preprocessor, model, evaluator):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model
        self.evaluator = evaluator

    def run(self, data=None):
        if data is None:
            data = self.data_loader.load()
        processed_data = self.preprocessor.process(data)
        self.model.fit(processed_data)
        result = self.evaluator.evaluate(self.model, processed_data)
        return result