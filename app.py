import gradio as gr
import pandas as pd
import pickle
import numpy as np


with open("pipeline.pkl","rb") as file:
  pipeline = pickle.load(file)


def predict_sales(na, eu, jp, other, year):
    input_data = pd.DataFrame([{
        "NA_Sales": na,
        "EU_Sales": eu,
        "JP_Sales": jp,
        "Other_Sales": other,
        "Year": year
    }])

    input_data = input_data.reindex(
        columns=pipeline.feature_names_in_,
        fill_value=0
    )

    return float(pipeline.predict(input_data)[0])


    
   
    # input_datas = pd.DataFrame([[na, eu, jp, other, year]],
    #     columns=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Year"]
    # )


    # input_data = input_datas.reindex(columns=X.columns, fill_value=0)
      
    # predition = model.predict(input_data)[0]
    # return predition

demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="NA Sales"),
        gr.Number(label="EU Sales"),
        gr.Number(label="JP Sales"),
        gr.Number(label="Other Sales"),
        gr.Number(label="Release Year")
    ],
    outputs=gr.Number(label="Predicted Global Sales"),
    title="Video Game Global Sales Predictor"
)

demo.launch()