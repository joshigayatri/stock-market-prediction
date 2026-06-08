import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- LOGIN USERS ----------------

USERS = {
    "admin": "admin123",
    "gayatri": "12345"
}

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#1C1F26;
}

div[data-testid="metric-container"]{
    background-color:#1C1F26;
    padding:15px;
    border-radius:15px;
    border:1px solid #2E3440;
}

.stButton>button{
    width:100%;
    background-color:#00C853;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------

if not st.session_state.logged_in:

    st.markdown("""
    <h1 style='text-align:center'>
    📈 Stock Market Prediction System
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.subheader("🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if username in USERS and USERS[username] == password:

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error("Invalid Username or Password")

    st.stop()

# ---------------- DASHBOARD ----------------

st.title("📈 Stock Market Prediction System")
st.markdown(
    "Analyze stocks and predict future prices using Machine Learning"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Navigation")

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.rerun()

stock_symbol = st.sidebar.selectbox(
    "Select Company",
    [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "WIPRO.NS",
        "LT.NS"
    ]
)

future_days = st.sidebar.slider(
    "Prediction Days",
    1,
    30,
    7
)

# ---------------- PREDICT BUTTON ----------------

if st.button("Predict"):

    data = yf.download(
        stock_symbol,
        start="2020-01-01",
        end="2025-01-01"
    )

    if data.empty:

        st.error("No Data Found")
        st.stop()

    close_data = data["Close"]

    if hasattr(close_data, "columns"):
        close_data = close_data.iloc[:, 0]

    current_price = float(close_data.iloc[-1])

    # ML MODEL

    df = pd.DataFrame()

    df["Close"] = close_data.values
    df["Day"] = range(len(df))

    X = df[["Day"]]
    y = df["Close"]

    model = LinearRegression()

    model.fit(X, y)

    accuracy = r2_score(
        y,
        model.predict(X)
    )

    future_day = pd.DataFrame(
        {
            "Day": [len(df) + future_days]
        }
    )

    predicted_price = model.predict(
        future_day
    )[0]

    change_percent = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    # COMPANY INFO

    try:

        ticker = yf.Ticker(stock_symbol)

        info = ticker.info

    except:

        info = {}

    # TABS

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Dashboard",
            "🤖 Prediction",
            "📈 Charts",
            "🏢 Company Info"
        ]
    )

    # ---------------- TAB 1 ----------------

    with tab1:

        st.subheader("📊 Market Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Current Price",
                f"₹ {current_price:.2f}"
            )

        with col2:

            st.metric(
                f"Predicted Price ({future_days} Days)",
                f"₹ {predicted_price:.2f}"
            )

        with col3:

            st.metric(
                "Expected Change",
                f"{change_percent:.2f}%"
            )

        with col4:

            st.metric(
                "Model Accuracy",
                f"{accuracy*100:.2f}%"
            )

        st.subheader("📋 Historical Data")

        st.dataframe(
            data.tail(10),
            use_container_width=True
        )

    # ---------------- TAB 2 ----------------

    with tab2:

        st.subheader("🤖 Prediction Result")

        st.metric(
            "Predicted Price",
            f"₹ {predicted_price:.2f}"
        )

        if change_percent > 2:

            st.success("🟢 BUY SIGNAL")

        elif change_percent < -2:

            st.error("🔴 SELL SIGNAL")

        else:

            st.warning("🟡 HOLD")

        st.write(
            f"Expected Change: {change_percent:.2f}%"
        )

    # ---------------- TAB 3 ----------------

    with tab3:

        chart_df = pd.DataFrame(
            {
                "Date": data.index,
                "Close": close_data.values
            }
        )

        fig = px.line(
            chart_df,
            x="Date",
            y="Close",
            title=f"{stock_symbol} Historical Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        df["Predicted"] = model.predict(X)

        fig2 = px.line(
            df,
            y=["Close", "Predicted"],
            title="Actual vs Predicted"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ---------------- TAB 4 ----------------

    with tab4:

        st.subheader("🏢 Company Information")

        st.write(
            "**Company:**",
            info.get("longName", "N/A")
        )

        st.write(
            "**Sector:**",
            info.get("sector", "N/A")
        )

        st.write(
            "**Industry:**",
            info.get("industry", "N/A")
        )

        st.write(
            "**Market Cap:**",
            info.get("marketCap", "N/A")
        )

# ---------------- FOOTER ----------------

st.markdown("---")

st.info(
    "This prediction is based on Linear Regression and is for educational purposes only."
)