import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

# ---------------- THEME ----------------
if theme == "Dark":
    st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("scaled.joblib")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center'>📩 Spam Message Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Machine Learning based Spam Detection</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- INPUT ----------------
st.subheader("✍️ Enter Message")
message = st.text_area(
    "",
    height=150,
    placeholder="Congratulations! You have won a free gift..."
)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict", use_container_width=True):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        X = vectorizer.transform([message])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        ham_prob = round(proba[0] * 100, 2)
        spam_prob = round(proba[1] * 100, 2)

        # Save history
        st.session_state.history.append({
            "Message": message,
            "Prediction": prediction.upper(),
            "Spam %": spam_prob,
            "Ham %": ham_prob
        })

        st.markdown("---")

        # ---------------- RESULT ----------------
        if prediction == "spam":
            st.error("🚨 **SPAM MESSAGE DETECTED**")
        else:
            st.success("✅ **HAM (NOT SPAM)**")

        col1, col2 = st.columns(2)
        col1.metric("Spam Probability", f"{spam_prob}%")
        col2.metric("Ham Probability", f"{ham_prob}%")

        # ---------------- BAR CHART ----------------
        st.subheader("📊 Confidence Bar Chart")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Ham", "Spam"], [ham_prob, spam_prob])
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Probability (%)")
        st.pyplot(fig1)

        # ---------------- PIE CHART ----------------
        st.subheader("🥧 Confidence Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie(
            [ham_prob, spam_prob],
            labels=["Ham", "Spam"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("🧾 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared!")
else:
    st.info("No predictions yet.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | ML Spam Classifier")
