
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from scipy.stats import norm

st.set_page_config(page_title="Tokopedia DSS Restock", layout="wide")

st.markdown("""
<style>
.main {background-color:#0e1117;color:white;}
div[data-testid="metric-container"]{background:#1f2937;padding:10px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("produk_tokopedia.csv")

    rename = {
        "Nama Produk":"product_name",
        "Nama Toko":"store_name",
        "Lokasi Toko":"city",
        "Terjual":"sold",
        "Jumlah Ulasan":"review_count",
        "Rating":"rating",
        "Harga (IDR)":"price",
        "Diskon (%)":"discount",
        "Produk URL":"product_url"
    }
    df = df.rename(columns=rename)

    def parse_sold(x):
        x = str(x).lower()
        if "rb" in x:
            try:
                return int(float(x.split("rb")[0].replace("+","").strip())*1000)
            except:
                return 0
        digits=''.join(c for c in x if c.isdigit())
        return int(digits) if digits else 0

    df["sold"] = df["sold"].apply(parse_sold)

    df["review_count"] = pd.to_numeric(
        df["review_count"].astype(str).str.extract(r'(\d+)')[0],
        errors="coerce"
    ).fillna(0)

    for c in ["rating","price","discount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["price"] = df["price"].fillna(df["price"].median())
    df["discount"] = df["discount"].fillna(0)

    return df.fillna("Unknown")

df = load_data()

st.sidebar.header("DSS Configuration")
ph = st.sidebar.slider("High Demand",0.0,1.0,0.5)
pm = st.sidebar.slider("Medium Demand",0.0,1.0,0.3)
pl = st.sidebar.slider("Low Demand",0.0,1.0,0.2)

risk = st.sidebar.selectbox("Risk Preference",
["Conservative","Moderate","Aggressive"])

mc_iter = st.sidebar.slider("Monte Carlo Iteration",1000,10000,3000)

st.title("Tokopedia DSS Restock Dashboard")

scaler = MinMaxScaler()

# Certainty
cert = scaler.fit_transform(df[["sold","rating","review_count"]])
df["certainty_score"] = cert[:,0]*0.5 + cert[:,1]*0.3 + cert[:,2]*0.2

# EV
df["ev_score"] = ph*(df["sold"]*1.2)+pm*(df["sold"])+pl*(df["sold"]*0.7)

# Uncertainty
payoff = pd.DataFrame({
"High":df["sold"]*1.2,
"Medium":df["sold"],
"Low":df["sold"]*0.7
})

df["maximax"]=payoff.max(axis=1)
df["maximin"]=payoff.min(axis=1)
df["laplace"]=payoff.mean(axis=1)

regret = payoff.max()-payoff
df["minimax_regret"]=-regret.max(axis=1)

unc = scaler.fit_transform(
df[["maximax","maximin","laplace","minimax_regret"]])
df["uncertainty_score"]=unc.mean(axis=1)

# Logistic Regression
threshold = df["sold"].quantile(0.75)
df["high_demand"]=(df["sold"]>=threshold).astype(int)

X=df[["rating","review_count","price","discount"]]
y=df["high_demand"]

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

pred=model.predict(X_test)
prob=model.predict_proba(X)[:,1]

df["probability_high_demand"]=prob

acc=accuracy_score(y_test,pred)
auc=roc_auc_score(y_test,model.predict_proba(X_test)[:,1])

# Utility
if risk=="Conservative":
    df["utility_score"]=np.sqrt(prob)
elif risk=="Moderate":
    df["utility_score"]=prob
else:
    df["utility_score"]=(prob**2)+df["certainty_score"]

# Monte Carlo
mu=max(df["sold"].mean(),1)
sigma=max(df["sold"].std(),1)

sim=np.random.normal(mu,sigma,mc_iter)

df["simulation_score"]=norm.cdf(df["sold"],mu,sigma)

# DSS
def normcol(c):
    return MinMaxScaler().fit_transform(df[[c]])[:,0]

df["final_dss_score"]=(
normcol("certainty_score")*0.15+
normcol("ev_score")*0.20+
normcol("uncertainty_score")*0.15+
normcol("probability_high_demand")*0.20+
normcol("utility_score")*0.15+
normcol("simulation_score")*0.15
)

rank=df.sort_values("final_dss_score",ascending=False)

st.header("Executive Overview")

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Total Produk",len(df))
c2.metric("Total Seller",df["store_name"].nunique())
c3.metric("Top Rating",round(df["rating"].max(),2))
c4.metric("Top Sold",int(df["sold"].max()))
c5.metric("Top DSS",round(rank.iloc[0]["final_dss_score"],4))

st.plotly_chart(px.bar(
df.nlargest(10,"sold"),
x="sold",y="product_name",
orientation="h"),use_container_width=True)

a,b=st.columns(2)
with a:
    st.plotly_chart(px.histogram(df,x="price"),use_container_width=True)
with b:
    st.plotly_chart(px.histogram(df,x="rating"),use_container_width=True)

city=df["city"].value_counts().head(15).reset_index()
st.plotly_chart(px.bar(city,x="city",y="count"),
use_container_width=True)

st.header("Decision Under Certainty")
st.dataframe(rank[["product_name","certainty_score"]].head(10))

st.header("Decision Under Risk")
st.dataframe(rank[["product_name","ev_score"]].head(10))

st.header("Decision Under Uncertainty")
st.dataframe(rank[[
"product_name","maximax","maximin",
"laplace","minimax_regret"]].head(20))

st.header("Probabilistic Modeling")
m1,m2=st.columns(2)
m1.metric("Accuracy",round(acc,4))
m2.metric("ROC-AUC",round(auc,4))

cm=confusion_matrix(y_test,pred)
st.plotly_chart(px.imshow(cm,text_auto=True),
use_container_width=True)

st.dataframe(rank[[
"product_name","probability_high_demand"]].head(10))

st.header("Utility & Risk Preference")
st.dataframe(rank[[
"product_name","utility_score"]].head(10))

st.header("Simulation & Sensitivity")

s1,s2,s3,s4=st.columns(4)
s1.metric("Expected Demand",int(sim.mean()))
s2.metric("Best Case",int(np.percentile(sim,95)))
s3.metric("Worst Case",int(np.percentile(sim,5)))
s4.metric("Prob High Demand",
f"{(sim>threshold).mean():.2%}")

st.plotly_chart(px.histogram(sim,nbins=50),
use_container_width=True)

st.header("DSS Integration")
st.dataframe(rank[[
"product_name","final_dss_score"]].head(10))

st.header("Final Recommendation")

best=rank.iloc[0]

st.success(f"""
Produk Terbaik : {best['product_name']}

DSS Score : {best['final_dss_score']:.4f}

Rating : {best['rating']}

Sold : {int(best['sold'])}

Probability High Demand : {best['probability_high_demand']:.2%}

Alasan:
Produk memiliki kombinasi skor certainty,
expected value, probabilitas permintaan,
utility dan simulasi terbaik.
""")
