%%writefile app.py

import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing
from openai import OpenAI
#import openai
import base64
import json
import os
from urllib.parse import urlparse

st.set_page_config(layout="wide")

os.environ["OPENAI_API_KEY"]= openai_token
client = OpenAI()


#----------------- GLobal variables -------------------------
image_folder_path= 'sample_data/'
#------------------------------------------------------------

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


i_input_type= st.selectbox('Choose task:', ["Open camera", "Upload image"])

if i_input_type == "Open camera": 
    i_prompt_template= st.selectbox("Choose prompt", ["Answer MCQ", "None"])
    if i_prompt_template == "Answer MCQ":
        i_mcq_type= st.radio("Choose MCQ type", ["Single", "Multiple"], horizontal=True)

        i_user_prompt= '''Extract all text from the images. The image contains text in the form of multiple choice questions.
         Ignore any watermarks. Format the output in the form of multiple choice questions. 
         provide suitable line breaks with numbering if any.'''

        if i_mcq_type == "Single":
            i_user_prompt+= "\n Each question can have only one answer."
        elif i_mcq_type == "Multiple":
            i_user_prompt+= "\n Each question can have multiple answers."
        
    elif i_prompt_template == "None":
        i_user_prompt= st.text_area("Type your prompt", "")
    
    image_local = st.camera_input("Take a picture")
    if image_local and len(i_prompt_template)>5:
        with open(os.path.join(image_folder_path,"test.jpg"),"wb") as f:
            f.write(image_local.getbuffer())

        image_local_temp = os.path.join(image_folder_path,"test.jpg")
        image_url = f"data:image/jpeg;base64,{encode_image(image_local_temp)}"

        

        response = client.chat.completions.create(
          model='gpt-4-vision-preview',
          messages=[
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": i_user_prompt},
                      {
                          "type": "image_url",
                          "image_url": {"url": image_url}
                      }
                  ],
              }
          ],
          max_tokens=1000,
      )
        
        ocr_string = response.choices[0].message.content
        st.write(ocr_string)

        i_user_prompt_final= '''Provide the correct answern for below multiple choice question.
          First answer what the correct answern is and then explain why you chose this in 2 to 3 lines. \n''' + ocr_string

        response= client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "user", "content": i_user_prompt_final}
          ]
      )
        st.divider()
        llm_output= response.choices[0].message.content
        llm_tokens= response.usage.total_tokens
        st.write(llm_output)
        st.metric(label="Tokens", value=llm_tokens)


