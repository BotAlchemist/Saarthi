

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
#from datetime import datetime  # Date and time operations
from datetime import datetime, date, time, timedelta
from dateutil.parser import parse
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Saarthi - Kuber",
    page_icon="🪙"  # You can use emojis like 🪙, 📊, 💵, 📉, etc.
)

# @st.cache_data(show_spinner=False)
# def load_data():
#   return pd.read_csv(FILE_PATH)

# FILE_PATH = 'Saarthi_Kuber.csv'

# Initialize OpenAI API key and passcode
i_key = 'sk-proj-iajReYBKXd12Kw4n2F0xXvwP7fHbEowI2O4fT2EkOZaQE2jydpOsxKR4coOWxgS9D1x9W-7IA6T3BlbkFJlPZ1iv2rmgoezm6-EuX-LDGMa21UoC9tWkZphZpQUBPKFCyn-gZAKuiMF3zF2fsyTPbKPGA60'
i_passcode = st.sidebar.text_input("OpenAI key", type='password')

# --------------------------- Firebase Setup ---
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_credentials"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()


# --- Load Data from Firestore ---
def load_kuber_data():
    docs = db.collection("expenses").stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    return df

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

    i_page_option= st.selectbox('Page', ['Kuber', 'Reminder', 'Diet'])

    if i_page_option== 'Kuber':
      i_menu_option= st.sidebar.selectbox('Menu', ['Add expense',  'Analyse' , 'Edit sheet'])
    
      if i_menu_option == 'Add expense':
          st.markdown("### Say something!")
          audio = audiorecorder("Click to record", "Click to stop recording")
          manual_text = st.text_input("Or type your expense:", placeholder="e.g., Bought mangoes 2kg 240Rs")
    
          if manual_text.strip():
              i_context = manual_text
          elif len(audio) > 2000:  # ensure audio is not too short (~0.1s)
              i_context = fetch_audio(audio)
          else:
              st.warning("Audio too short or no input provided. Please enter text or record longer audio.")
              st.stop()
    
          st.divider()
          st.write(i_context)
    
          i_final_prompt = '''Context: {}'''.format(i_context)
          i_final_prompt += '''Date: {}'''.format(datetime.now())
          i_final_prompt += i_prompt
    
          i_chat_prompt = '''You are a helpful financial advisor/assistant for Indian household. Only generate the json object and not any explanation. '''
          response = client.chat.completions.create(
              model="gpt-4o",
              messages=[
                  {"role": "system", "content": i_chat_prompt},
                  {"role": "user", "content": i_final_prompt}
              ]
          )
    
          i_response = str(response.choices[0].message.content).replace('''```json''', '').replace('''```''', '')
          new_expense = json.loads(i_response)
    
          editable_expense = {}
          current_date = datetime.now().date()
    
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
    
          st.write('Extracted values')
          editable_expense_record = pd.DataFrame([editable_expense])
          st.write(editable_expense_record)
    
          if st.button('Add Expense'):
              data_to_add = editable_expense.copy()
              data_to_add["Date"] = data_to_add["Date"].strftime('%d-%m-%Y')
              db.collection("expenses").add(data_to_add)
              ai_voice("'Expense Added!'")

      elif i_menu_option== 'Edit sheet':
        docs = db.collection("expenses").stream()
        data = []
        for doc in docs:
            record = doc.to_dict()
            record["doc_id"] = doc.id
            data.append(record)
        df = pd.DataFrame(data)

        if 'Date' in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')

        edited_df = st.data_editor(df.drop(columns=["doc_id"]), num_rows="dynamic", use_container_width=True)

        if st.button("Save Changes"):
            for idx, row in edited_df.iterrows():
                original = df.loc[idx]
                doc_id = original["doc_id"]

                updated_data = {}
                for col in edited_df.columns:
                    if not pd.isna(row[col]) and row[col] != original[col]:
                        value = row[col]
                        if isinstance(value, pd.Timestamp):
                            value = value.strftime('%d-%m-%Y')
                        updated_data[col] = value

                if updated_data:
                    db.collection("expenses").document(doc_id).update(updated_data)
            st.success("Changes saved to Firestore!")
  
      elif i_menu_option== 'Analyse':
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()

        df = load_kuber_data()

        if df.empty:
            st.warning("No expense data found in Firestore.")
        else:
            st.markdown("Analyze your expenses deeply with smart visuals and KPIs 👇")

            st.sidebar.header("🔎 Filter Your Data")
            all_categories = df["Category"].dropna().unique().tolist()
            selected_categories = st.sidebar.multiselect("Select Category", options=all_categories, default=all_categories)

            filtered_df = df[df["Category"].isin(selected_categories)]

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

            st.subheader("📈 Monthly Spending Trend")
            df_monthly = filtered_df.groupby(filtered_df["Date"].dt.to_period("M"))['Amount'].sum().reset_index()
            df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
            st.line_chart(df_monthly.set_index("Date"))

            st.subheader("📅 Last 30 Days Expense")
            last_30 = filtered_df[filtered_df["Date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
            last_30_daily = last_30.groupby("Date")["Amount"].sum().reset_index()
            st.line_chart(last_30_daily.set_index("Date"))

            st.subheader("📊 Spending by Category")
            category_totals = filtered_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
            st.bar_chart(category_totals)

            st.subheader("📆 Monthly Expense by Type")
            df_type_month = filtered_df.copy()
            df_type_month["Month"] = df_type_month["Date"].dt.to_period("M").dt.to_timestamp()
            df_grouped = df_type_month.groupby(["Month", "Type"])["Amount"].sum().unstack().fillna(0)
            st.bar_chart(df_grouped)

            st.subheader("🥇 Top 10 Items by Spend")
            top_items = filtered_df.groupby("Item")["Amount"].sum().sort_values(ascending=False).head(10)
            st.bar_chart(top_items)

            st.subheader("⚖️ Cost Efficiency (₹ per Unit)")
            avg_cost = filtered_df.groupby("Item")["CostPerQuantity"].mean().sort_values(ascending=False)
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=avg_cost.values, y=avg_cost.index, ax=ax1)
            ax1.set_xlabel("₹ per Unit")
            st.pyplot(fig1)

            st.subheader("🔍 Needs vs Wants Spending")
            type_summary = filtered_df.groupby("Type")["Amount"].sum()
            st.bar_chart(type_summary)

            st.subheader("📉 Wants Budget Utilization")
            income = st.number_input("Enter your monthly income (₹):", min_value=1000, max_value=1000000, value=90000, step=500)
            wants_budget = 0.3 * income
            spent_on_wants = filtered_df[filtered_df["Type"] == "Wants"]["Amount"].sum()
            remaining_wants = max(wants_budget - spent_on_wants, 0)
            
            st.markdown(f"**Wants Budget (30%):** ₹{wants_budget:,.0f}")
            
            col1, col2 = st.columns(2)
            col1.metric("Spent on Wants", f"₹{spent_on_wants:,.0f}")
            col2.metric("Remaining Wants Budget", f"₹{remaining_wants:,.0f}")
            
            fig2, ax2 = plt.subplots(figsize=(5, 0.5))
            ax2.barh(["Wants Budget"], [spent_on_wants], color='salmon', label="Spent")
            ax2.barh(["Wants Budget"], [remaining_wants], left=[spent_on_wants], color='lightgreen', label="Remaining")
            ax2.set_xlim(0, wants_budget)
            ax2.set_xlabel("Amount (₹)")
            ax2.legend(loc="upper right")
            st.pyplot(fig2)

           

            st.markdown("---")
            st.caption("Built with ❤️ by SAARTHI-Kuber")

#_____________________________________ DIET PAGE ____________________________
    elif  i_page_option== 'Diet':
        import diet  # A diet.py file
        diet.run(i_passcode)

#_______________________________________ REMINDER PAGE ___________________________________

    elif  i_page_option== 'Reminder':
        # import reminder
        # reminder.run()
      
      
      CATEGORIES = ['Bills', 'Maintenance', 'Housing', 'Rent', 'Insurance', 'Financial', 'Personal', 'Appointment']
      FREQUENCIES = ['One-time', 'Daily', 'Weekly', 'Every 28 Days', 'Monthly', 'Yearly']
      
      # --- Data Loading / Persistence ---
      def load_data():
          reminders_ref = db.collection("reminders")
          docs = reminders_ref.stream()
          records = []
          for doc in docs:
              record = doc.to_dict()
              record['id'] = doc.id
              record['due'] = pd.to_datetime(record['due'])
              records.append(record)
          return pd.DataFrame(records)
      
      def save_reminder_to_db(title, due, category, frequency, completed=False):
          db.collection("reminders").add({
              'title': title,
              'due': due.isoformat(),
              'category': category,
              'frequency': frequency,
              'completed': completed
          })
      
      def update_reminder_status(doc_id, completed=True):
          db.collection("reminders").document(doc_id).update({'completed': completed})
      
      # --- Helper Functions ---
      def add_reminder(title: str, due: datetime, category: str, frequency: str):
          save_reminder_to_db(title, due, category, frequency)
          st.success(f"Added {frequency} reminder: '{title}' @ {due}")
      
      def next_due(current: datetime, frequency: str):
          if frequency == 'Daily':
              return current + timedelta(days=1)
          if frequency == 'Weekly':
              return current + timedelta(weeks=1)
          if frequency == 'Every 28 Days':
              return current + timedelta(days=28)
          if frequency == 'Monthly':
              return current + timedelta(days=30)
          if frequency == 'Yearly':
              try:
                  return current.replace(year=current.year+1)
              except:
                  return current + timedelta(days=365)
          return None
      
     
      
      # Sidebar: Add Reminder Form
      st.sidebar.header("Add New Reminder")
      with st.sidebar.form(key='add_form'):
          text = st.text_input("What to remind?", placeholder="Pay Wi-Fi bill on 5th every month")
          dt = st.date_input("Date", value=date.today())
          tm = st.time_input("Time", value=time(hour=9, minute=0))
          cat = st.selectbox("Category", CATEGORIES)
          freq = st.selectbox("Frequency", FREQUENCIES)
          submitted = st.form_submit_button("Add Reminder")
          if submitted and text:
              try:
                  due_dt = parse(text, default=datetime.combine(dt, tm))
              except:
                  due_dt = datetime.combine(dt, tm)
              add_reminder(text, due_dt, cat, freq)
      
      # Load DataFrame
      df = load_data()
      
      # Define Tabs: Today, All, Upcoming
      tab1, tab2, tab3 = st.tabs(["Today's Agenda", "All Reminders", "Upcoming Reminders"])
      
      # --- Today's Agenda ---
      with tab1:
          st.subheader("Due Today")
          today = datetime.now().date()
          todays = df[(df['due'].dt.date == today) & (~df['completed'])].sort_values('due')
          if not todays.empty:
              for _, row in todays.iterrows():
                  col1, col2 = st.columns([5,1])
                  with col1:
                      time_str = row['due'].strftime('%H:%M') if pd.notna(row['due']) else 'Unknown'
                      st.markdown(f"**{row['title']}** [{row['frequency']}] – {time_str}")
                  with col2:
                      if st.button("Done", key=f"done_{row['id']}"):
                          update_reminder_status(row['id'], completed=True)
                          nxt = next_due(row['due'], row['frequency'])
                          if nxt:
                              add_reminder(row['title'], nxt, row['category'], row['frequency'])
                          st.success("Marked as done!")
          else:
              st.info("No reminders for today. 🎉")
      
      # --- All Reminders ---
      with tab2:
          st.subheader("All Reminders")
          col1, col2, col3, col4 = st.columns([3,3,2,2])
          with col1:
              cats = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
          with col2:
              freqs = st.multiselect("Frequency", FREQUENCIES, default=FREQUENCIES)
          with col3:
              start = st.date_input("From", value=df['due'].dt.date.min() if df['due'].notna().any() else date.today())
          with col4:
              end = st.date_input("To", value=df['due'].dt.date.max() if df['due'].notna().any() else date.today())
          df['due_date'] = df['due'].dt.date
          view = df[
              df['category'].isin(cats) &
              df['frequency'].isin(freqs) &
              df['due_date'].between(start, end)
          ].sort_values(['due', 'completed'])
      
          if not view.empty:
              table = view[['title', 'due', 'category', 'frequency', 'completed']].copy()
              table['due'] = table['due'].dt.strftime('%Y-%m-%d %H:%M')
              st.dataframe(table, use_container_width=True)
          else:
              st.info("No reminders match your filters.")
      
      # --- Upcoming Reminders ---
      with tab3:
          st.subheader("Upcoming Reminders")
          now = datetime.now()
          upcoming = df[(df['due'] > now) & (~df['completed'])].sort_values('due')
          if not upcoming.empty:
              table_up = upcoming[['title', 'due', 'category', 'frequency']].copy()
              table_up['due'] = table_up['due'].dt.strftime('%Y-%m-%d %H:%M')
              table_up['days_left'] = (upcoming['due'] - now).apply(lambda x: f"{x.days} days left")
              st.dataframe(table_up, use_container_width=True)
          else:
              st.info("No upcoming reminders!")
      
      # Footer
      st.markdown("---")
      st.write("Built with ❤️ by SmartReminder")


