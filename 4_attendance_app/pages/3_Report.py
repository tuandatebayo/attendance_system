import streamlit as st
from Home import face_rec

st.set_page_config(page_title="Report", layout="wide")
st.subheader("Report")

#Retriving logs data and show it
#extract data from redis list
name = 'attendance:logs'
def load_logs(name, end = -1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list

#tabs show info
tab1, tab2 = st.tabs(['Logs', 'Data'])

with tab1:
    if st.button('Refresh logs'):
        st.write(load_logs(name))
with tab2:
    if st.button('Refresh data'):
        with st. spinner("Retriving Data from Redis database ..."):
            redis_face_db = face_rec.retrieve_data(name='academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])
