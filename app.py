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

#i_key= 'sk-proj-aVAzex4cFCRIU0kIqZWT3BlbkFJF2wZ0WEuG7themYfcubn'
i_key= 'sk-proj-gUo7UuBh5llI5FHenFKjT3BlbkFJ01MwxYNzCtIQD9t426H'
i_passcode = st.sidebar.text_input("OpenAI key", type='password')

if len(i_passcode) > 0:
    #insertion_index= 11
    #i_key= i_key[:insertion_index] + i_passcode + i_key[insertion_index:]
    i_key = i_key + i_passcode
    #----------------- Global variables -------------------------
    image_folder_path= 'sample_data/'
    os.environ["OPENAI_API_KEY"]= i_key
    client = OpenAI()
    #------------------------------------------------------------

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_text_from_image(i_user_prompt, image_url ):
        response = client.chat.completions.create(
              model='gpt-4o',
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
        return response.choices[0].message.content

    def get_gpt_response(i_user_prompt_final, i_temperature, i_model):
        response= client.chat.completions.create(
              model=i_model,
              messages=[
                  {"role": "user", "content": i_user_prompt_final}
              ],
              temperature=i_temperature
          )
        return response.choices[0].message.content, response.usage.total_tokens


    i_menu= st.sidebar.selectbox("Menu", ['Chat', 'Vision', 'Co-pilot'])
    i_openai_model= st.radio('Choose model: ',['gpt-3.5-turbo', 'gpt-4o'] , horizontal=True)

#------------------------------------- Use GTP chat ---------------------------------------------    
    if i_menu== 'Chat':
        i_chat_prompt= st.text_area(":writing_hand:", placeholder="Type your prompt", height=200, key='chat_key')
        i_temperature = st.slider(":thermometer:", min_value = 0.0, max_value = 2.0, value= 0.3, step=0.1)
        
        got_response= False
        if st.button("Ask") and len(i_chat_prompt)>5:
            st.divider()
            llm_output, llm_tokens= get_gpt_response(i_chat_prompt, i_temperature, i_openai_model)
            got_response= True
        
        if got_response:
            st.write(llm_output)
            st.divider()
            st.metric(label="Tokens", value=llm_tokens)
    

#------------------------------------- Use GTP vision ---------------------------------------------
    elif i_menu== 'Vision':   
        i_input_type= st.selectbox('Choose task:', ["Open camera", "Upload image"])

        if i_input_type == "Open camera":
            i_prompt_template= st.selectbox("Choose prompt", ["Answer","Label", "None"])
            if i_prompt_template == "Answer":
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

            elif i_prompt_template == "Label":
                i_user_prompt= '''You are provided with a image of product with its labels and ingredients list.
                Your goal is to extract all texts from labels, ingredients or nutrition facts..
               
                If the image doesnot contain any labels or ingredients, provide a response: "Image does not contain necessary details".
                '''

            image_local = st.camera_input("Take a picture")
            if image_local and len(i_user_prompt)>5:
                with open(os.path.join(image_folder_path,"test.jpg"),"wb") as f:
                    f.write(image_local.getbuffer())

                image_local_temp = os.path.join(image_folder_path,"test.jpg")
                image_url = f"data:image/jpeg;base64,{encode_image(image_local_temp)}"

                #st.write(i_user_prompt)
                ocr_string = extract_text_from_image(i_user_prompt, image_url )
                st.write(ocr_string)

                if i_prompt_template == "Answer":
                    i_user_prompt_final= '''Provide the correct answern for below multiple choice question.
                      First answer what the correct answern is and then explain why you chose this in 2 to 3 lines. \n''' + ocr_string
                    st.divider()
                    llm_output, llm_tokens= get_gpt_response(i_user_prompt_final, 0.2, i_openai_model)
                    st.write(llm_output)
                    st.metric(label="Tokens", value=llm_tokens)

                elif i_prompt_template == "Label":
                    i_user_prompt_final=''' You are an expert nutritionist.
                    Carefully analyze  all the labels and ingredients.
                    Provide the list of ingredients along with a line of what the ingredient is.
                    Provide a brief summary of the extracted label information, highlighting the key details.
                    Focus more on the additives, preservatives, artificial colors or flavors or any harmful substances present.
                    ''' + ocr_string

                    st.divider()
                    llm_output, llm_tokens= get_gpt_response(i_user_prompt_final, 0.3)
                    st.write(llm_output)
                    st.metric(label="Tokens", value=llm_tokens)

                    




else:
    st.info("Please enter your passcode.")
    
    # st.code("sk-proj-7uK5yZ4zEeXyPbrMPJf3sdOrpVHgyEsAHGig94MGVzW1AxdRXF")
    # st.code("sk-proj-nugHpvIH1whBPpcEVLnktMHfQTNh7n2muDQRrM5wd6DTNsYlJz")
    # st.code("sk-proj-aVA9zex4cECRIU1kIqZWT3BlbkFJF2wZ0WEuG7tpemSfxubn")
    # st.code("sk-proj-c4uc7o2F5VGsSYgc1PfUgDtAE6KNC8iMrJRZKVz32Kh0N1Olb3")
    # st.code("sk-proj-ITf7c0lWVCeNi2DPU3YWobQTAn6evVQlnN9Z7f8pDquTQuVhv")
    # st.code("sk-proj-nugHpvIH1whBPpcEVLnktMHfQTNh7n2muDQRrM5wd6DTNsYlJj")
    # st.code("sk-proj-5lNHypFjNexYEkqjNawyXRl0dlR8FNiVjd6GxoLyAtan5ZtXx")
    # st.code("sk-proj-S4svUFupfUHlH5XRU6nbCuwKuS5E8fhka8Ub3EfkpW7d5QZn")
    # st.code("sk-proj-7uK5yZ4zEeXyPbrMPJf3sdOrpVHgyEsAHGig94MGVzW1Axdr")
    # st.code("sk-proj-aVA7zex4cFCRIU0kIqZWT3BlbkFJF2wZ0WEuG7thenYfcubn")
    # st.code("sk-proj-FIfvIkWdKghp9qaCR7XlLU9EoMu6iYjoSeDVtL3BRtO7pUbo")
