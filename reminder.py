import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date, time
from dateutil.parser import parse
from google.cloud import firestore  # Assuming you're using Firestore for DB

# Initialize Firestore DB
db = firestore.Client()


def run():
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
