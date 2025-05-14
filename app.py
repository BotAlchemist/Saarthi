# Import necessary libraries
import streamlit as st  # Web App framework
import pandas as pd  # Data manipulation
from PIL import Image  # Image processing
import numpy as np  # Numerical operations
from openai import OpenAI  # OpenAI API
import openai
import base64
import json
import os
import base64
from urllib.parse import urlparse
from audiorecorder import audiorecorder  # Audio recording
from datetime import datetime  # Date and time operations

# Set Streamlit page configuration
st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
  return pd.read_csv(FILE_PATH)

FILE_PATH = '/content/drive/MyDrive/Saarthi/Kuber/Saarthi_Kuber.csv'

# Initialize OpenAI API key and passcode
i_key = 'sk-proj-gUo7UuBh5llI5FHenFKjT3BlbkFJ01MwxYNzCtIQD9t426H'
i_passcode = st.sidebar.text_input("OpenAI key", type='password')

# Define categories and types for expenses
category_list = ['Fruit and Vegetable', 'Dairy', 'Meat and Egg', 'Grocery', 'Gas', 'Driver', 'Bills', 'House', 'EMI', 'Entertainment', 'Travel', 'Food and Snacks', 'Healthcare', 'Education', 'Personal Care', 'Savings', 'Miscellaneous']
type_list = ['Needs', 'Wants', 'Savings', 'Loan repayment']

# Define the prompt for OpenAI API
i_prompt = ''' You are to return a json string extracting information from the provided context. Ensure there is no extra data or characters before or after the JSON object that might be causing the parser to fail."

{"Date":,
  "Item": ,
  "Category": ,
  "Amount": ,
  "Quantity": ,
  "Unit": ,
  "CostPerQuantity": ,
  "Type": ,
  "Comment":}

guildelines:
Everything should be in English language.
Date in DD-MM-YYYY format.
Extract item, amount, quantity, from context provided.
Identify which category Item belongs to from the list:
'Fruit and Vegetable', 'Dairy', 'Meat and Egg', 'Grocery', 'Gas', 'Driver', 'Bills', 'House', 'EMI', 'Entertainment', 'Travel', 'Food and Snacks', 'Healthcare', 'Education', 'Personal Care', 'Savings', 'Miscellaneous'.


Quantity should only have integer value like : 10, 200 etc
identify unit from the context. Can have values like : "gm", "kg", "nos" (if in quantity number).
Calculate cost per quantity in gram/price, litre/price and quantity/price. if unit is kg, always convert to gm first.
Type: 'Needs', 'Wants', 'Savings', 'Loan repayment'.
Comment: if any, else write 'N/A'

Sample example:
Context: Carrot 200 gm , 500 Rs.

{"Date": "12-05-2025",
  "Item": "Carrot",
  "Category": "Vegetable",
  "Amount": 500,
  "Quantity": 200,
  "Unit": "gm",
  "CostPerQuantity": 2.5,
  "Type": "Needs",
  "Comment": "N/A"}
'''


# ------------------------------------------------------ Function definition
def fetch_audio(audio):
  # Save audio to a file
  audio.export("audio.wav", format="wav")

  # Initialize OpenAI client
  client = OpenAI()

  # Open and transcribe the audio file
  audio_file = open("audio.wav", "rb")
  transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
  )

  return transcription.text

# Function to return AI voice
def ai_voice(i_text_input):
  text_input = i_text_input

  voice = "shimmer"  #st.selectbox("Choose a voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
  model = "tts-1"  #st.selectbox("Choose quality", ["tts-1", "tts-1-hd"])

  response = openai.audio.speech.create(
                model=model,
                voice=voice,
                input=text_input
            )

  # Save to temporary audio file
  audio_path = "ai_voice.mp3"
  with open(audio_path, "wb") as f:
    f.write(response.content)

  # Read and encode to base64
  with open(audio_path, "rb") as f:
      audio_bytes = f.read()
      b64 = base64.b64encode(audio_bytes).decode()

  # Embed autoplay audio in HTML
  st.markdown(
      f"""
      <audio autoplay controls>
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          Your browser does not support the audio element.
      </audio>
      """,
      unsafe_allow_html=True
  )


# Check if OpenAI passcode is entered
if len(i_passcode) <= 0:
    st.info("Please enter your OpenAI key")
else:
    # Combine base key with the user-provided passcode
    i_key = i_key + i_passcode

    # Set OpenAI API key in environment
    os.environ["OPENAI_API_KEY"] = i_key
    client = OpenAI()

    i_menu_option= st.sidebar.selectbox('Menu', ['Add expense',  'Analyse' , 'Edit sheet'])

    #---------------------------- Add Expense option -----------------------------
    if i_menu_option== 'Add expense':
      # Streamlit UI for recording audio
      st.markdown("### Say something!")
      audio = audiorecorder("Click to record", "Click to stop recording")

      # Process audio if recorded
      if len(audio) > 0:
          i_context= fetch_audio(audio)

          st.divider()

          # Display transcribed text
          st.write(i_context)

          # Prepare the context and prompt for OpenAI API

          i_final_prompt = '''Context: {}'''.format(i_context)
          i_final_prompt += '''Date: {}'''.format(datetime.now())
          i_final_prompt += i_prompt

          # Define chat prompt for OpenAI API
          i_chat_prompt = '''You are a helpful financial advisor/assistant for Indian household. Only generate the json object and not any explanation. '''
          response = client.chat.completions.create(
              model="gpt-4o",
              messages=[
                  {"role": "system", "content": i_chat_prompt},
                  {"role": "user", "content": i_final_prompt}
              ]
          )

          # Extract and clean the response
          i_response = str(response.choices[0].message.content).replace('''```json''', '').replace('''```''', '')
          new_expense = json.loads(i_response)

          # Create editable fields for each key in the dictionary
          editable_expense = {}

          # Get the current date
          current_date = datetime.now().date()

          # Create input fields for each key in the expense dictionary
          for key, value in new_expense.items():
              if key == 'Date':
                  editable_expense[key] = st.date_input(f"{key}:", value=current_date)
              elif key in ['Item', 'Unit', 'Comment']:
                  editable_expense[key] = st.text_input(f"{key}:", value=value)
              elif key == 'Category':
                  if value not in category_list:
                      category_list.append(value)
                  editable_expense[key] = st.selectbox(f"{key}:", options=category_list, index=category_list.index(value))
              elif key in ['Amount', 'Quantity', 'CostPerQuantity']:
                  editable_expense[key] = st.number_input(f"{key}:", value=value)
              elif key == 'Type':
                  if value not in type_list:
                      type_list.append(value)
                  editable_expense[key] = st.selectbox(f"{key}:", options=type_list, index=type_list.index(value))
              else:
                  pass

          # Display the editable dictionary
          st.write('Extracted values')

          # Convert the editable dictionary to a DataFrame
          editable_expense_record = pd.DataFrame([editable_expense])
          st.write(editable_expense_record)

          # Button to add expense to CSV
          if st.button('Add Expense'):
              # Append the new record DataFrame to the CSV
              editable_expense_record.to_csv(FILE_PATH, mode='a', header=False, index=False)
              #st.success('Expense Added!')
              ai_voice("'Expense Added!'")

    #---------------------------- Edit sheet option -----------------------------
    elif i_menu_option== 'Edit sheet':
      df= pd.read_csv(FILE_PATH)
      #st.write(df)

      # Show editable table
      edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

      # Button to save changes
      if st.button("Save Changes to CSV"):
          edited_df.to_csv(FILE_PATH, index=False)
          st.success("Changes saved successfully!")

    #---------------------------- Analyse option -----------------------------
    elif i_menu_option== 'Analyse':

      if st.button("🔄 Refresh Data"):
        st.cache_data.clear()




      import matplotlib.pyplot as plt
      import seaborn as sns

      # Load your expense data
      df = load_data()
      df["Date"] = pd.to_datetime(df["Date"])


      st.markdown("Analyze your expenses deeply with smart visuals and KPIs 👇")

      # ---- FILTERS ----
      st.sidebar.header("🔎 Filter Your Data")

      # Category filter
      all_categories = df["Category"].unique().tolist()
      selected_categories = st.sidebar.multiselect("Select Category", options=all_categories, default=all_categories)

      # # Date filter
      # min_date = df["Date"].min()
      # max_date = df["Date"].max()
      # selected_date = st.sidebar.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

      # Apply filters
      filtered_df = df[df["Category"].isin(selected_categories)]

      # ---- KPI SECTION ----
      st.subheader("📌 Key Performance Indicators")
      col1, col2, col3, col4 = st.columns(4)

      with col1:
          st.metric("Total Expense", f"₹ {filtered_df['Amount'].sum():,.2f}")
      with col2:
          st.metric("Average Daily Spend", f"₹ {filtered_df.groupby('Date')['Amount'].sum().mean():.2f}")
      with col3:
          top_cat = filtered_df.groupby("Category")["Amount"].sum()
          st.metric("Top Spending Category", top_cat.idxmax() if not top_cat.empty else "N/A")
      with col4:
          needs_sum = filtered_df[filtered_df["Type"] == "Needs"]["Amount"].sum()
          wants_sum = filtered_df[filtered_df["Type"] == "Wants"]["Amount"].sum() if "Wants" in filtered_df["Type"].unique() else 0
          st.metric("Needs vs Wants (₹)", f"{needs_sum:.0f} / {wants_sum:.0f}")

      # ---- TIME TREND ----
      st.subheader("📈 Monthly Spending Trend")
      df_monthly = filtered_df.groupby(filtered_df["Date"].dt.to_period("M"))['Amount'].sum().reset_index()
      df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
      st.line_chart(df_monthly.set_index("Date"))

      # ---- CATEGORY BREAKDOWN ----
      st.subheader("📊 Spending by Category")
      category_totals = filtered_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
      st.bar_chart(category_totals)

      # ---- TYPE BY MONTH ----
      st.subheader("📆 Monthly Expense by Type")
      df_type_month = filtered_df.copy()
      df_type_month["Month"] = df_type_month["Date"].dt.to_period("M").dt.to_timestamp()
      df_grouped = df_type_month.groupby(["Month", "Type"])["Amount"].sum().unstack().fillna(0)
      st.bar_chart(df_grouped)

      # ---- TOP ITEMS ----
      st.subheader("🥇 Top 10 Items by Spend")
      top_items = filtered_df.groupby("Item")["Amount"].sum().sort_values(ascending=False).head(10)
      st.bar_chart(top_items)

      # ---- COST EFFICIENCY ----
      st.subheader("⚖️ Cost Efficiency (₹ per Unit)")
      avg_cost = filtered_df.groupby("Item")["CostPerQuantity"].mean().sort_values(ascending=False)
      fig1, ax1 = plt.subplots(figsize=(8, 4))
      sns.barplot(x=avg_cost.values, y=avg_cost.index, ax=ax1)
      ax1.set_xlabel("₹ per Unit")
      st.pyplot(fig1)

      # ---- NEEDS VS WANTS ----
      st.subheader("🔍 Needs vs Wants Spending")
      type_summary = filtered_df.groupby("Type")["Amount"].sum()
      st.bar_chart(type_summary)

      # ---- WEEKDAY SPENDING ----
      st.subheader("📅 Average Spending by Day of Week")
      filtered_df["Weekday"] = filtered_df["Date"].dt.day_name()
      weekday_avg = filtered_df.groupby("Weekday")["Amount"].mean().reindex(
          ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
      st.bar_chart(weekday_avg)


      st.markdown("---")
      st.caption("Built with ❤️ by SAARTHI-Kuber")

