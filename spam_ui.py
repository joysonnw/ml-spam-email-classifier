import streamlit as st
import pandas as pd
from spam_filter import train_model

model, vectorizer, accuracy, report_data = train_model(True)

st.title("Spam Meassage Classifier")
if accuracy:    
    st.write(f"**Model Accuracy:** {accuracy:.2%}")
    report_df = pd.DataFrame(report_data).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)

user_input = st.text_area("Enter a message to classify:")

if st.button("Classify"):
    if user_input.strip():
        msg_vec = vectorizer.transform([user_input])
        prediction = model.predict(msg_vec)[0]
        if prediction == "spam":
            st.error("This message is a SPAM!")
        else:
            st.success("This message is a HAM.")
    else:
        st.warning("Please enter a message.")
        
