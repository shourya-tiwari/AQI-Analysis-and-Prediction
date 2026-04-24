from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pune_aqi.data.cleaning import PuneAqiSchema, clean_pune_aqi_dataset  # noqa: E402
from pune_aqi.data.loaders import DataPaths, load_default_dataset  # noqa: E402
from pune_aqi.models.train import results_to_frame, train_and_evaluate_models  # noqa: E402


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "neon.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_dataset(csv_path: str) -> pd.DataFrame:
    if not csv_path.strip():
        return load_default_dataset(repo_root=REPO_ROOT)
    df = pd.read_csv(csv_path)
    return clean_pune_aqi_dataset(df, schema=PuneAqiSchema())


@st.cache_data(show_spinner=False)
def _train_metrics(df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    results = train_and_evaluate_models(df, test_size=test_size, random_state=random_state)
    return results_to_frame(results, split="test")


def main() -> None:
    st.set_page_config(page_title="Pune AQI Predictor", layout="wide", initial_sidebar_state="expanded")
    _load_css()

    st.markdown(
        """
        <div class="hero">
          <div class="hero__title">Pune AQI Prediction Lab</div>
          <div class="hero__subtitle">Neon dashboard for AQI regression model comparison.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    schema = PuneAqiSchema()
    paths = DataPaths(repo_root=REPO_ROOT)

    with st.sidebar:
        st.markdown("### Dataset")
        csv_path = st.text_input("CSV path", value=str(paths.default_csv))
        st.caption(f"Default: `{paths.default_csv.as_posix()}`")

        st.markdown("### Training")
        test_size = st.slider("Test size", 0.10, 0.40, 0.25, 0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=0, step=1)

    df = _load_dataset(csv_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Cities", f"{df[schema.city].nunique():,}" if schema.city in df.columns else "NA")
    c3.metric("Stations", f"{df[schema.station].nunique():,}" if schema.station in df.columns else "NA")
    latest_aqi = float(df[schema.aqi].dropna().iloc[-1]) if schema.aqi in df.columns and len(df) else float("nan")
    c4.metric("Latest AQI", f"{latest_aqi:.0f}" if pd.notna(latest_aqi) else "NA")

    leaderboard, explorer = st.columns([1.1, 1])

    with leaderboard:
        st.markdown("### Model leaderboard (test split)")
        metrics_df = _train_metrics(df, float(test_size), int(random_state))
        st.dataframe(
            metrics_df.style.format({"r2": "{:.4f}", "rmse": "{:.3f}", "mae": "{:.3f}", "rmsle": "{:.4f}"}),
            use_container_width=True,
        )

        best_model = metrics_df.index[0] if len(metrics_df) else None
        if best_model:
            st.markdown(
                f"""
                <div class="callout">
                  <div class="callout__title">Best model</div>
                  <div class="callout__value">{best_model}</div>
                  <div class="callout__hint">Sorted by lowest RMSE on the test split.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with explorer:
        st.markdown("### AQI distribution")
        if schema.aqi in df.columns and df[schema.aqi].notna().any():
            fig = px.histogram(df, x=schema.aqi, nbins=40, title="AQI histogram", template="plotly_dark")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No AQI values found in this dataset.")

        st.markdown("### Recent rows (cleaned)")
        st.dataframe(df.tail(25), use_container_width=True)

    st.markdown("### Run locally")
    st.code("streamlit run app/streamlit_app.py", language="bash")

    # Stop here to prevent legacy code below from running.
    st.stop()


if __name__ == "__main__":
    main()


# --- Legacy script below (kept for reference; not executed) ---
if False:
    import numpy as np
import pandas as pd

df = pd.read_csv("data set\\state_weather_aqi_data_mf2.csv")

x1 = df.iloc[:,:12].values
z1 = pd.DataFrame(x1)

y1 = df.iloc[:,12:13].values
z2 = pd.DataFrame(y1)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
x_new1 = pd.DataFrame(ohe.fit_transform(x1[:,[0]]).toarray()) #state
x_new2 = pd.DataFrame(ohe.fit_transform(x1[:,[1]]).toarray()) #city
x_new3 = pd.DataFrame(ohe.fit_transform(x1[:,[2]]).toarray()) #station

feature_set = pd.concat([x_new1,x_new2,x_new3,pd.DataFrame(x1[:,5:12])],axis=1,sort=False)

# importing ml libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

x_train,x_test,y_train,y_test = train_test_split(feature_set,y1,test_size=0.25,random_state=0)

#-----------------------------------------------
#---- test data prediction ---------------------
#-----------------------------------------------

# multiple linear regression model
mreg = LinearRegression()
mreg.fit(x_train,y_train)

mlr_y_predict = mreg.predict(x_test)

# ---------------------------------------------
# polynomial regression model
# degree = 2

poly_reg = PolynomialFeatures(degree = 2)
preg = LinearRegression()
pf = poly_reg.fit_transform(x_train)
preg.fit(pf,y_train)

pr_y_predict = preg.predict(poly_reg.fit_transform(x_test))

#-----------------------------------------------
# decision tree regression model

dec_tree = DecisionTreeRegressor(random_state = 0)
dec_tree.fit(x_train,y_train)

dt_y_predict = dec_tree.predict(x_test)

#-----------------------------------------------
# random forest regression model
# random forest with 500 trees

rt_reg = RandomForestRegressor(n_estimators = 500, random_state = 0)
rt_reg.fit(x_train,y_train)

rt_y_predict = rt_reg.predict(x_test)

#-----------------------------------------------
# support vector regression model

# --- feature scaling the paramenters for better results ---
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_svr = sc_x.fit_transform(x_train)
y_train_svr = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_svr,y_train_svr)

svr_y_predict = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_test)).reshape(1,-1))

#----------------------------------------------
# error estimation methods
#----------------------------------------------

from math import sqrt
from sklearn import metrics

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0:
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return ((sum/len(predicted))**0.5)[0]

#----- multiple linear regresion -------
rmse_mlr = sqrt(metrics.mean_squared_error(y_test, mlr_y_predict))
mae_mlr = metrics.mean_absolute_error(y_test, mlr_y_predict)
r2_mlr = metrics.r2_score(y_test,mlr_y_predict)
rmsle_mlr = rmsle(y_test,mlr_y_predict)

#----- polynomial regression ------------ 
rmse_pr = sqrt(metrics.mean_squared_error(y_test, pr_y_predict))
mae_pr = metrics.mean_absolute_error(y_test, pr_y_predict)
r2_pr = metrics.r2_score(y_test,pr_y_predict)
rmsle_pr = rmsle(y_test,pr_y_predict)

#----- decision tree regression ---------
rmse_dt = sqrt(metrics.mean_squared_error(y_test, dt_y_predict))
mae_dt = metrics.mean_absolute_error(y_test, dt_y_predict)
r2_dt = metrics.r2_score(y_test,dt_y_predict)
rmsle_dt = rmsle(y_test,dt_y_predict)

#----- random forest regression ---------
rmse_rt = sqrt(metrics.mean_squared_error(y_test, rt_y_predict))
mae_rt = metrics.mean_absolute_error(y_test, rt_y_predict)
r2_rt = metrics.r2_score(y_test,rt_y_predict)
rmsle_rt = rmsle(y_test,rt_y_predict)

#----- support vextor regression --------
rmse_svr = sqrt(metrics.mean_squared_error(y_test, svr_y_predict.T))
mae_svr = metrics.mean_absolute_error(y_test, svr_y_predict.T)
r2_svr = metrics.r2_score(y_test,svr_y_predict.T)
rmsle_svr = rmsle(y_test,svr_y_predict.T)

'''
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
fs = sc_x.fit_transform(feature_set)
cvs = cross_val_score(mreg,fs,y1,cv=20)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
'''
#------------------------------------
#---- training data prediction ------
#------------------------------------

# ---- MLR ------
mlr_ytp_rmse = sqrt(metrics.mean_squared_error(y_train, mreg.predict(x_train)))
mlr_ytp_mae = metrics.mean_absolute_error(y_train, mreg.predict(x_train))
mlr_ytp_r2 = metrics.r2_score(y_train, mreg.predict(x_train))
m1 = mreg.predict(x_train)
mlr_ytp_rmsle = rmsle(y_train, m1)
#------ polynomial regression ---------
pr_ytp_rmse = sqrt(metrics.mean_squared_error(y_train, preg.predict(poly_reg.fit_transform(x_train))))
pr_ytp_mae = metrics.mean_absolute_error(y_train, preg.predict(poly_reg.fit_transform(x_train)))
pr_ytp_r2 = metrics.r2_score(y_train, preg.predict(poly_reg.fit_transform(x_train)))
pr_ytp_rmsle = rmsle(y_train, preg.predict(poly_reg.fit_transform(x_train)))

#mxp = preg.predict(poly_reg.fit_transform(x_train))

# ----- decision tree reg ------
dt_ytp_rmse = sqrt(metrics.mean_squared_error(y_train, dec_tree.predict(x_train)))
dt_ytp_mae = metrics.mean_absolute_error(y_train, dec_tree.predict(x_train))
dt_ytp_r2 = metrics.r2_score(y_train, dec_tree.predict(x_train))
dt_ytp_rmsle = rmsle(y_train, dec_tree.predict(x_train))

# ----- random forest reg -----
rf_ytp_rmse = sqrt(metrics.mean_squared_error(y_train, rt_reg.predict(x_train)))
rf_ytp_mae = metrics.mean_absolute_error(y_train, rt_reg.predict(x_train))
rf_ytp_r2 = metrics.r2_score(y_train, rt_reg.predict(x_train))
rf_ytp_rmsle = rmsle(y_train, rt_reg.predict(x_train))

# ----- svr -----
svr_ytp_rmse = sqrt(metrics.mean_squared_error(y_train, sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_train)).reshape(1,-1)).T)) 
svr_ytp_mae = metrics.mean_absolute_error(y_train, sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_train)).reshape(1,-1)).T)  
svr_ytp_r2 = metrics.r2_score(y_train, sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_train)).reshape(1,-1)).T) 
svr_ytp_rmsle = rmsle(y_train, sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_train)).reshape(1,-1)).T)  

# ==========================================
# =========== RESULT =======================
# ==========================================

print("")
print("evaluating on training data:")
print("---------------------------------")
print("models\tR^2\tRMSE\tMAE\tRMSLE")
print("MLR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(mlr_ytp_r2,mlr_ytp_rmse,mlr_ytp_mae,mlr_ytp_rmsle))
print("PR\t{0:.2f}\t{1:.2f}\t{2:.3f}\t{3:.4f}".format(pr_ytp_r2,pr_ytp_rmse,pr_ytp_mae,pr_ytp_rmsle))
print("DTR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(dt_ytp_r2,dt_ytp_rmse,dt_ytp_mae,dt_ytp_rmsle))
print("RFR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(rf_ytp_r2,rf_ytp_rmse,rf_ytp_mae,rf_ytp_rmsle))
print("SVR\t{0:.4f}\t{1:.3f}\t{2:.3f}\t{3:.4f}".format(svr_ytp_r2,svr_ytp_rmse,svr_ytp_mae,svr_ytp_rmsle))
print("")
print("evaluating on testing data:")
print("---------------------------------")
print("models\tR^2\tRMSE\tMAE\tRMSLE")
print("MLR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(r2_mlr,rmse_mlr,mae_mlr,rmsle_mlr))
print("PR\t{0:.2f}\t{1:.2f}\t{2:.3f}\t{3:.4f}".format(r2_pr,rmse_pr,mae_pr,rmsle_pr))
print("DTR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(r2_dt,rmse_dt,mae_dt,rmsle_dt))
print("RFR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(r2_rt,rmse_rt,mae_rt,rmsle_rt))
print("SVR\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(r2_svr,rmse_svr,mae_svr,rmsle_svr))

