import streamlit as st
import os
import sys


# ================= PATH SETUP =================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.predict import predict_news



# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="📰",
    layout="wide"
)



# ================= CUSTOM CSS =================

st.markdown(
"""
<style>


.block-container {

    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;

}


/* HERO */

.hero {

background: linear-gradient(
135deg,
#2563eb,
#9333ea
);

padding: 45px;

border-radius: 25px;

text-align:center;

box-shadow:
0px 10px 35px rgba(147,51,234,0.3);

margin-bottom:30px;

}


.hero h1 {

font-size:50px;

font-weight:900;

color:white;

}


.hero p {

font-size:20px;

color:#e5e7eb;

}


/* Cards */


.card {

background:#111827;

padding:25px;

border-radius:20px;

box-shadow:
0 5px 20px rgba(0,0,0,.25);

}



/* Metric */


[data-testid="stMetric"] {

background:#111827;

padding:20px;

border-radius:18px;

box-shadow:
0 5px 15px rgba(0,0,0,.3);

}



/* Buttons */

.stButton button {

height:50px;

font-size:17px;

font-weight:bold;

border-radius:15px;

background:
linear-gradient(
135deg,
#2563eb,
#9333ea
);

color:white;

border:none;

}


/* Text area */


textarea {

font-size:16px !important;

border-radius:15px !important;

}



/* Result */


.result {

padding:35px;

border-radius:25px;

text-align:center;

font-size:40px;

font-weight:900;

margin-top:30px;

}


.real {

background:#064e3b;

color:#4ade80;

}


.fake {

background:#7f1d1d;

color:#f87171;

}


/* footer */

.footer {

text-align:center;

color:#9ca3af;

margin-top:50px;

}


</style>

""",
unsafe_allow_html=True
)





# ================= SIDEBAR =================


with st.sidebar:


    st.title(
        "📰 TruthGuard AI"
    )


    st.write(
        """
        An AI powered Fake News Detection
        system built using NLP and Machine Learning.
        """
    )


    st.divider()


    st.subheader(
        "⚙️ Technology"
    )


    st.write(
        """

        🐍 Python

        📚 NLTK

        📊 TF-IDF Vectorizer

        🤖 Linear SVM

        🚀 Streamlit

        """
    )


    st.divider()



    st.subheader(
        "📈 Performance"
    )


    st.metric(
        "Accuracy",
        "99.36%"
    )


    st.metric(
        "Dataset",
        "44,898"
    )


    st.divider()


    st.info(
        "Built for NLP Binary Text Classification"
    )





# ================= HERO =================


st.markdown(
"""

<div class='hero'>

<h1>📰 TruthGuard AI</h1>

<p>
Detect Fake News using Natural Language Processing & Machine Learning
</p>

</div>

""",
unsafe_allow_html=True
)





# ================= TOP METRICS =================


m1,m2,m3 = st.columns(3)


m1.metric(
    "🎯 Accuracy",
    "99.36%"
)


m2.metric(
    "📰 Articles",
    "44K+"
)


m3.metric(
    "🤖 Model",
    "Linear SVM"
)




st.write("")





# ================= MAIN AREA =================


left,right = st.columns(
    [2,1]
)



with left:


    st.subheader(
        "🔎 Analyze News Article"
    )


    example = st.button(
        "✨ Load Example News"
    )


    sample=""


    if example:


        sample = """
WASHINGTON (Reuters) - Government officials announced
a new policy after a formal discussion.
"""



    news = st.text_area(
        "Paste News Content",
        value=sample,
        height=300,
        placeholder="Paste complete news article here..."
    )



    analyze = st.button(
        "🚀 Verify Authenticity",
        use_container_width=True
    )




with right:


    st.markdown(
    """

<div class='card'>

<h3>⚙️ Detection Pipeline</h3>

<br>

✅ Text Cleaning

<br><br>

✅ Stopword Removal

<br><br>

✅ TF-IDF Extraction

<br><br>

✅ ML Prediction


</div>


""",
unsafe_allow_html=True
)






# ================= PREDICTION =================


if analyze:


    if news.strip()=="":


        st.warning(
            "⚠️ Please enter a news article."
        )


    else:


        prediction, confidence = predict_news(
            news
        )



        if prediction == 1:


            st.markdown(
            """

<div class='result real'>

✅ REAL NEWS

</div>

""",
unsafe_allow_html=True
)


        else:


            st.markdown(
            """

<div class='result fake'>

🚨 FAKE NEWS

</div>

""",
unsafe_allow_html=True
)




        st.subheader(
            "Prediction Confidence"
        )


        st.progress(
            confidence/100
        )


        st.success(
            f"{confidence}% Confidence"
        )





st.divider()





# ================= MODEL DETAILS =================


st.subheader(
    "📊 Model Information"
)


c1,c2 = st.columns(2)



with c1:


    st.markdown(
    """

<div class='card'>

<h3>Machine Learning</h3>

Algorithm: Linear SVM

Feature Extraction: TF-IDF

Accuracy: 99.36%

</div>


""",
unsafe_allow_html=True
)



with c2:


    st.markdown(
    """

<div class='card'>

<h3>NLP Processing</h3>

Lowercasing

Stopword Removal

Stemming

Text Cleaning

</div>


""",
unsafe_allow_html=True
)





# ================= ABOUT =================


with st.expander(
    "📌 About This Project"
):


    st.write(
        """
        TruthGuard AI is an end-to-end Machine Learning
        project designed to classify news articles as
        Fake or Real.

        The system was trained on the Fake and Real News Dataset
        containing more than 44,000 articles.

        The pipeline includes text preprocessing,
        TF-IDF feature extraction and Linear SVM classification.
        """
    )






# ================= FOOTER =================


st.markdown(
"""

<div class='footer'>

🚀 Built using Python | NLP | Machine Learning | Streamlit

</div>

""",
unsafe_allow_html=True
)