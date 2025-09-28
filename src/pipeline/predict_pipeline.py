# pylint: skip-file

import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, load_config

config = load_config()

predict_config = config["predict_pipeline"]
model_path = predict_config["model_path"]
preprocessor_path = predict_config["preprocessor_path"]

class PredictPipeline:
    def predict(self, features):
        try:
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred_result = model.predict(data_scaled)
            return pred_result
        except Exception as e:
            raise CustomException(e, None)


class CustomData:
    """
    responsible for taking the inputs that we are giving in the HTML to the backend
    """

    def __init__(
        self,
        Time: float,
        V1: float, V2: float, V3: float, V4: float, V5: float, V6: float, V7: float, V8: float, V9: float, V10: float,
        V11: float, V12: float, V13: float, V14: float, V15: float, V16: float, V17: float, V18: float, V19: float, V20: float,
        V21: float, V22: float, V23: float, V24: float, V25: float, V26: float, V27: float, V28: float,
        Amount: float
    ):
        self.Time = Time
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount


    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                "Time": [self.Time],
                "V1": [self.V1],
                "V2": [self.V2],
                "V3": [self.V3],
                "V4": [self.V4],
                "V5": [self.V5],
                "V6": [self.V6],
                "V7": [self.V7],
                "V8": [self.V8],
                "V9": [self.V9],
                "V10": [self.V10],
                "V11": [self.V11],
                "V12": [self.V12],
                "V13": [self.V13],
                "V14": [self.V14],
                "V15": [self.V15],
                "V16": [self.V16],
                "V17": [self.V17],
                "V18": [self.V18],
                "V19": [self.V19],
                "V20": [self.V20],
                "V21": [self.V21],
                "V22": [self.V22],
                "V23": [self.V23],
                "V24": [self.V24],
                "V25": [self.V25],
                "V26": [self.V26],
                "V27": [self.V27],
                "V28": [self.V28],
                "Amount": [self.Amount]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, None)
    
if __name__ == "__main__":
    # Example: Provide dummy values for all required features
    data = CustomData(
        Time=0.0,
        V1=0.0, V2=0.0, V3=0.0, V4=0.0, V5=0.0, V6=0.0, V7=0.0, V8=0.0, V9=0.0, V10=0.0,
        V11=0.0, V12=0.0, V13=0.0, V14=0.0, V15=0.0, V16=0.0, V17=0.0, V18=0.0, V19=0.0, V20=0.0,
        V21=0.0, V22=0.0, V23=0.0, V24=0.0, V25=0.0, V26=0.0, V27=0.0, V28=0.0,
        Amount=0.0
    )
    pred_df = data.get_data_as_dataframe()
    print("Input DataFrame:")
    print(pred_df)
    predict_pipeline = PredictPipeline()
    pred_result = predict_pipeline.predict(pred_df)
    print("Prediction result:", pred_result)

    output_dir = os.path.join("src", "prediction_output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prediction.txt"), "w") as f:
        f.write(str(pred_result))

