import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# ------------------ Load Model ------------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("scaled.joblib")

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
    .main {
        background-color: #f9fafb;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("<div class='title'>📩 Spam Message Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a message is Spam or Ham using ML</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Input Section ------------------
st.subheader("✍️ Enter Message")
message = st.text_area(
    "Type your message below:",
    height=150,
    placeholder="Congratulations! You won a free prize..."
)

# ------------------ Prediction ------------------
if st.button("🔍 Predict", use_container_width=True):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform input
        X_input = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(X_input)[0]

        # Probability
        proba = model.predict_proba(X_input)[0]
        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        st.markdown("---")

        # ------------------ Result ------------------
        if prediction == "spam":
            st.error(f"🚨 **SPAM DETECTED**")
        else:
            st.success(f"✅ **HAM (Not Spam)**")

        # ------------------ Probability Display ------------------
        col1, col2 = st.columns(2)

        col1.metric("📌 Spam Probability", f"{spam_prob:.2f}%")
        col2.metric("📌 Ham Probability", f"{ham_prob:.2f}%")

        # ------------------ Bar Chart ------------------
        st.subheader("📊 Prediction Confidence")

        labels = ['Ham', 'Spam']
        values = [ham_prob, spam_prob]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🚀 Built with Streamlit & Machine Learning")
st.caption("© 2024 Spam Message Detector")
