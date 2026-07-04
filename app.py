"""
Shopper Spectrum — Customer Analytics & Recommendation System
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ----------------------------------------------------------------------
if "k_clusters" not in st.session_state:
    st.session_state.k_clusters = 4

# ----------------------------------------------------------------------
# CACHED HELPER FUNCTIONS
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, encoding="latin1")
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    data = df.copy()
    report = {}
    report["original_rows"] = len(data)

    # 1. Missing values (CustomerID / Description)
    missing_before = data.isnull().sum()
    data = data.dropna(subset=["CustomerID"])
    report["rows_after_missing_removal"] = len(data)
    report["missing_summary"] = missing_before

    # 2. Duplicates
    dup_count = data.duplicated().sum()
    data = data.drop_duplicates()
    report["duplicates_removed"] = int(dup_count)

    # 3. Cancelled orders (InvoiceNo starting with 'C', if present)
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    cancelled = data["InvoiceNo"].str.startswith("C").sum()
    data = data[~data["InvoiceNo"].str.startswith("C")]
    report["cancelled_removed"] = int(cancelled)

    # 4. Invalid quantity / price
    invalid_qty_price = ((data["Quantity"] <= 0) | (data["UnitPrice"] <= 0)).sum()
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    report["invalid_qty_price_removed"] = int(invalid_qty_price)

    # Type fixes + derived column
    data["CustomerID"] = data["CustomerID"].astype(int)
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    data["Description"] = data["Description"].astype(str).str.strip()

    report["final_rows"] = len(data)
    return data, report


@st.cache_data(show_spinner=False)
def compute_rfm(data: pd.DataFrame):
    snapshot_date = data["InvoiceDate"].max() + timedelta(days=1)
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


@st.cache_data(show_spinner=False)
def scale_rfm(rfm: pd.DataFrame):
    # log transform to reduce skew, then standardize
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(
        lambda x: np.log1p(x.clip(lower=0))
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)
    return scaled, scaler


@st.cache_data(show_spinner=False)
def compute_elbow(scaled_rfm, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_rfm)
        inertias.append(km.inertia_)
    return list(k_range), inertias


def get_segment_labels(k: int):
    master = [
        "At-Risk", "Hibernating", "Occasional", "Need Attention",
        "Regular", "Loyal", "High-Value", "Champion",
    ]
    if k <= len(master):
        idx = np.linspace(0, len(master) - 1, k).round().astype(int)
        return [master[i] for i in idx]
    return [f"Segment {i+1}" for i in range(k)]


@st.cache_resource(show_spinner=False)
def run_segmentation(rfm: pd.DataFrame, k: int):
    scaled, scaler = scale_rfm(rfm)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(scaled)

    rfm_seg = rfm.copy()
    rfm_seg["Cluster"] = clusters

    # rank clusters by combined value score to assign meaningful labels
    cluster_profile = rfm_seg.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    cluster_profile["Score"] = (
        cluster_profile["Frequency"].rank()
        + cluster_profile["Monetary"].rank()
        - cluster_profile["Recency"].rank()
    )
    order = cluster_profile.sort_values("Score").index.tolist()
    labels = get_segment_labels(k)
    label_map = {cluster_id: labels[rank] for rank, cluster_id in enumerate(order)}

    rfm_seg["Segment"] = rfm_seg["Cluster"].map(label_map)
    return rfm_seg, km, scaler, label_map


@st.cache_data(show_spinner=False)
def build_similarity(data: pd.DataFrame):
    # description -> most common stockcode mapping (for display)
    desc_map = (
        data.groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )

    matrix = data.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0
    )
    binary_matrix = (matrix > 0).astype(int)

    sim = cosine_similarity(binary_matrix.T)
    sim_df = pd.DataFrame(sim, index=binary_matrix.columns, columns=binary_matrix.columns)
    return sim_df, desc_map


def recommend_products(product_name, sim_df, desc_map, top_n=5):
    # find stockcode(s) matching the description (case-insensitive)
    matches = [sc for sc, desc in desc_map.items() if desc.lower() == product_name.lower()]
    if not matches:
        return None
    stock_code = matches[0]
    if stock_code not in sim_df.columns:
        return None
    scores = sim_df[stock_code].drop(labels=[stock_code]).sort_values(ascending=False)
    top_codes = scores.head(top_n)
    results = pd.DataFrame({
        "StockCode": top_codes.index,
        "Product": [desc_map.get(sc, "Unknown") for sc in top_codes.index],
        "Similarity Score": top_codes.values.round(3),
    })
    return results


# ----------------------------------------------------------------------
# SIDEBAR — TITLE + NAVIGATION + DATA UPLOAD
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🛒 Shopper Spectrum")
st.sidebar.caption("Customer Analytics Dashboard")
st.sidebar.markdown("---")

PAGES = [
    "📘 Project Overview",
    "📊 Dashboard",
    "🧹 Data Cleaning",
    "📈 EDA & Insights",
    "🎯 RFM Analysis",
    "📐 Elbow Method",
    "👥 Customer Segmentation",
    "🔗 Product Similarity",
    "📦 Product Recommendation",
    "🔮 Customer Prediction",
    "📌 Final Insights",
]

st.sidebar.markdown("#### Navigation")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
st.sidebar.caption("Expected columns: InvoiceNo, StockCode, Description, "
                    "Quantity, InvoiceDate, UnitPrice, CustomerID, Country")

# ----------------------------------------------------------------------
# PROJECT OVERVIEW (works even without data)
# ----------------------------------------------------------------------
if page == "📘 Project Overview":
    st.markdown("# 🛒 Shopper Spectrum")
    st.markdown("## Customer Analytics & Recommendation System")
    st.markdown("---")

    st.markdown("### 🎯 Objective")
    st.write(
        "Analyze customer shopping behaviour, segment customers using RFM analysis, "
        "and recommend products using similarity analysis."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 Business Problems Solved")
        st.markdown(
            "- Who are the most valuable customers?\n"
            "- Which customers buy regularly vs. rarely?\n"
            "- Which customers have stopped shopping (at-risk)?\n"
            "- Which products sell the most?\n"
            "- What should be recommended to each customer?\n"
            "- Which country generates the highest revenue?"
        )
    with col2:
        st.markdown("#### 🛠️ Project Flow")
        st.markdown(
            "1. Load & understand dataset\n"
            "2. Clean data (missing, duplicates, invalid rows)\n"
            "3. Exploratory Data Analysis\n"
            "4. RFM feature engineering\n"
            "5. Elbow method → optimal clusters\n"
            "6. KMeans segmentation + labeling\n"
            "7. Item-based product similarity\n"
            "8. Interactive recommendation + prediction"
        )

    st.markdown("---")
    st.markdown("#### 📁 Dataset Columns")
    st.table(pd.DataFrame({
        "Column": ["InvoiceNo", "StockCode", "Description", "Quantity",
                   "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
        "Meaning": ["Bill number", "Product code", "Product name",
                    "Number of products purchased", "Purchase date and time",
                    "Price of one product", "Customer number", "Customer's country"],
    }))

    if uploaded_file is None:
        st.info("👈 Upload your CSV from the sidebar to activate every page.")

# ----------------------------------------------------------------------
# STOP HERE IF NO FILE UPLOADED (for all data-dependent pages)
# ----------------------------------------------------------------------
elif uploaded_file is None:
    st.warning("⚠️ Please upload a CSV dataset from the sidebar to continue.")
    st.stop()

else:
    raw_df = load_data(uploaded_file)
    clean_df, clean_report = clean_data(raw_df)

    # ------------------------------------------------------------------
    # DASHBOARD
    # ------------------------------------------------------------------
    if page == "📊 Dashboard":
        st.title("📊 Dashboard")
        st.markdown("High-level snapshot of the business after cleaning.")

        total_revenue = clean_df["TotalPrice"].sum()
        total_orders = clean_df["InvoiceNo"].nunique()
        total_customers = clean_df["CustomerID"].nunique()
        total_countries = clean_df["Country"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Total Revenue", f"£{total_revenue:,.0f}")
        c2.metric("🧾 Total Orders", f"{total_orders:,}")
        c3.metric("👥 Total Customers", f"{total_customers:,}")
        c4.metric("🌍 Countries", f"{total_countries}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            monthly = clean_df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
            fig = px.line(monthly, x="InvoiceDate", y="TotalPrice", markers=True,
                           title="Monthly Revenue Trend")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_countries = clean_df.groupby("Country")["TotalPrice"].sum().nlargest(10).reset_index()
            fig = px.bar(top_countries, x="TotalPrice", y="Country", orientation="h",
                         title="Top 10 Countries by Revenue")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # DATA CLEANING
    # ------------------------------------------------------------------
    elif page == "🧹 Data Cleaning":
        st.title("🧹 Data Cleaning")
        st.markdown("Steps applied to raw transaction data before analysis.")

        st.markdown("#### Missing Values (raw data)")
        st.dataframe(clean_report["missing_summary"].rename("Missing Count"))

        st.markdown("#### Cleaning Steps & Row Impact")
        steps = pd.DataFrame({
            "Step": [
                "Original rows",
                "After removing missing CustomerID",
                "Duplicate rows removed",
                "Cancelled orders removed (InvoiceNo starting with 'C')",
                "Invalid Quantity/UnitPrice (≤ 0) removed",
                "Final clean rows",
            ],
            "Value": [
                clean_report["original_rows"],
                clean_report["rows_after_missing_removal"],
                clean_report["duplicates_removed"],
                clean_report["cancelled_removed"],
                clean_report["invalid_qty_price_removed"],
                clean_report["final_rows"],
            ],
        })
        st.table(steps)

        pct_kept = clean_report["final_rows"] / clean_report["original_rows"] * 100
        st.success(f"✅ {pct_kept:.1f}% of original rows retained after cleaning.")

        st.markdown("#### Preview of Cleaned Data")
        st.dataframe(clean_df.head(20), use_container_width=True)

    # ------------------------------------------------------------------
    # EDA & INSIGHTS
    # ------------------------------------------------------------------
    elif page == "📈 EDA & Insights":
        st.title("📈 EDA & Insights")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Top Products", "Country-wise Sales", "Sales Trend", "Customer Behaviour"]
        )

        with tab1:
            top_products = (
                clean_df.groupby("Description")["Quantity"].sum().nlargest(15).reset_index()
            )
            fig = px.bar(top_products, x="Quantity", y="Description", orientation="h",
                         title="Top 15 Products by Quantity Sold", color="Quantity")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            country_sales = clean_df.groupby("Country").agg(
                Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique")
            ).reset_index().sort_values("Revenue", ascending=False)
            fig = px.pie(country_sales.head(10), names="Country", values="Revenue",
                         title="Revenue Share — Top 10 Countries")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(country_sales, use_container_width=True)

        with tab3:
            clean_df["Month"] = clean_df["InvoiceDate"].dt.to_period("M").astype(str)
            monthly_orders = clean_df.groupby("Month")["InvoiceNo"].nunique().reset_index()
            fig = px.line(monthly_orders, x="Month", y="InvoiceNo", markers=True,
                           title="Monthly Order Volume")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            active_customers = (
                clean_df.groupby("CustomerID")["InvoiceNo"].nunique()
                .sort_values(ascending=False).head(10).reset_index()
            )
            active_customers.columns = ["CustomerID", "Number of Orders"]
            fig = px.bar(active_customers, x="CustomerID", y="Number of Orders",
                         title="Top 10 Most Active Customers")
            fig.update_layout(xaxis_type="category")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # RFM ANALYSIS
    # ------------------------------------------------------------------
    elif page == "🎯 RFM Analysis":
        st.title("🎯 RFM Analysis")
        st.markdown(
            "**Recency** = days since last purchase · "
            "**Frequency** = number of distinct invoices · "
            "**Monetary** = total amount spent."
        )

        rfm = compute_rfm(clean_df)
        st.session_state["rfm"] = rfm

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
        c2.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
        c3.metric("Avg Monetary", f"£{rfm['Monetary'].mean():,.0f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(px.histogram(rfm, x="Recency", nbins=40, title="Recency Distribution"),
                             use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(rfm, x="Frequency", nbins=40, title="Frequency Distribution"),
                             use_container_width=True)
        with col3:
            st.plotly_chart(px.histogram(rfm, x="Monetary", nbins=40, title="Monetary Distribution"),
                             use_container_width=True)

        st.markdown("#### RFM Table")
        st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

    # ------------------------------------------------------------------
    # ELBOW METHOD
    # ------------------------------------------------------------------
    elif page == "📐 Elbow Method":
        st.title("📐 Elbow Method")
        st.markdown("Finding the optimal number of clusters (K) for KMeans segmentation.")

        rfm = st.session_state.get("rfm", compute_rfm(clean_df))
        scaled, _ = scale_rfm(rfm)
        k_values, inertias = compute_elbow(scaled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers"))
        fig.update_layout(title="Elbow Curve (Inertia vs. K)",
                           xaxis_title="Number of Clusters (K)", yaxis_title="Inertia (WCSS)")
        st.plotly_chart(fig, use_container_width=True)

        st.info("👀 Look for the 'elbow' point where inertia stops dropping sharply — "
                "that's usually the best K. Set the final K on the **Customer Segmentation** page.")

    # ------------------------------------------------------------------
    # CUSTOMER SEGMENTATION
    # ------------------------------------------------------------------
    elif page == "👥 Customer Segmentation":
        st.title("👥 Customer Segmentation")

        rfm = st.session_state.get("rfm", compute_rfm(clean_df))
        k = st.slider("Select number of clusters (K)", min_value=2, max_value=8,
                       value=st.session_state.k_clusters)
        st.session_state.k_clusters = k

        rfm_seg, km_model, scaler, label_map = run_segmentation(rfm, k)
        st.session_state["rfm_seg"] = rfm_seg
        st.session_state["km_model"] = km_model
        st.session_state["scaler"] = scaler
        st.session_state["label_map"] = label_map

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(
                rfm_seg, x="Recency", y="Monetary", size="Frequency", color="Segment",
                hover_data=["CustomerID"], title="Customer Segments (Recency vs Monetary)",
                log_y=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            seg_counts = rfm_seg["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig2 = px.pie(seg_counts, names="Segment", values="Count", title="Segment Share")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Segment Profile (average RFM per segment)")
        profile = rfm_seg.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
        profile["Customers"] = rfm_seg["Segment"].value_counts()
        st.dataframe(profile.sort_values("Monetary", ascending=False), use_container_width=True)

        st.markdown("#### Full Segmented Customer List")
        st.dataframe(rfm_seg.sort_values("Monetary", ascending=False), use_container_width=True)

    # ------------------------------------------------------------------
    # PRODUCT SIMILARITY
    # ------------------------------------------------------------------
    elif page == "🔗 Product Similarity":
        st.title("🔗 Product Similarity")
        st.markdown("Item-based collaborative filtering: products frequently bought "
                     "by the same customers are considered similar.")

        with st.spinner("Building similarity matrix..."):
            sim_df, desc_map = build_similarity(clean_df)
        st.session_state["sim_df"] = sim_df
        st.session_state["desc_map"] = desc_map

        st.success(f"✅ Similarity matrix built for {sim_df.shape[0]:,} products.")

        product_list = sorted(set(desc_map.values()))
        selected = st.selectbox("Pick a product to inspect its most similar items", product_list)
        results = recommend_products(selected, sim_df, desc_map, top_n=10)
        if results is not None:
            fig = px.bar(results, x="Similarity Score", y="Product", orientation="h",
                         title=f"Products most similar to '{selected}'")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(results, use_container_width=True)

    # ------------------------------------------------------------------
    # PRODUCT RECOMMENDATION
    # ------------------------------------------------------------------
    elif page == "📦 Product Recommendation":
        st.title("📦 Product Recommendation")
        st.markdown("Enter a product name to get similar product recommendations.")

        if "sim_df" not in st.session_state:
            with st.spinner("Building similarity matrix..."):
                sim_df, desc_map = build_similarity(clean_df)
            st.session_state["sim_df"] = sim_df
            st.session_state["desc_map"] = desc_map
        else:
            sim_df = st.session_state["sim_df"]
            desc_map = st.session_state["desc_map"]

        product_list = sorted(set(desc_map.values()))
        query = st.selectbox("🔍 Search / select a product", product_list)
        top_n = st.slider("Number of recommendations", 3, 10, 5)

        if st.button("Get Recommendations", type="primary"):
            results = recommend_products(query, sim_df, desc_map, top_n=top_n)
            if results is None:
                st.error("Product not found in similarity matrix.")
            else:
                st.markdown(f"#### Customers who bought **{query}** also bought:")
                for _, row in results.iterrows():
                    st.markdown(f"- **{row['Product']}**  (similarity: {row['Similarity Score']:.3f})")
                st.dataframe(results, use_container_width=True)

    # ------------------------------------------------------------------
    # CUSTOMER PREDICTION
    # ------------------------------------------------------------------
    elif page == "🔮 Customer Prediction":
        st.title("🔮 Customer Prediction")
        st.markdown("Enter a customer's RFM values to predict which segment they belong to.")

        if "km_model" not in st.session_state:
            st.warning("⚠️ Please visit **Customer Segmentation** page first to train the model.")
        else:
            km_model = st.session_state["km_model"]
            scaler = st.session_state["scaler"]
            label_map = st.session_state["label_map"]

            c1, c2, c3 = st.columns(3)
            recency = c1.number_input("Recency (days since last purchase)", min_value=0, value=30)
            frequency = c2.number_input("Frequency (number of orders)", min_value=0, value=5)
            monetary = c3.number_input("Monetary (total spend £)", min_value=0.0, value=500.0)

            if st.button("Predict Segment", type="primary"):
                input_log = np.log1p([[recency, frequency, monetary]])
                input_scaled = scaler.transform(input_log)
                cluster = km_model.predict(input_scaled)[0]
                segment = label_map[cluster]

                emoji_map = {
                    "Champion": "🏆", "High-Value": "💎", "Loyal": "🤝",
                    "Regular": "🙂", "Need Attention": "⚠️", "Occasional": "🕒",
                    "Hibernating": "😴", "At-Risk": "🚨",
                }
                emoji = emoji_map.get(segment, "👤")
                st.success(f"### {emoji} Predicted Segment: **{segment}**")

    # ------------------------------------------------------------------
    # FINAL INSIGHTS
    # ------------------------------------------------------------------
    elif page == "📌 Final Insights":
        st.title("📌 Final Insights & Recommendations")

        rfm_seg = st.session_state.get("rfm_seg")
        if rfm_seg is None:
            st.warning("⚠️ Please visit **Customer Segmentation** page first.")
        else:
            top_segment = rfm_seg.groupby("Segment")["Monetary"].sum().idxmax()
            biggest_segment = rfm_seg["Segment"].value_counts().idxmax()
            at_risk_count = (rfm_seg["Segment"].isin(["At-Risk", "Hibernating"])).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("💎 Highest Revenue Segment", top_segment)
            c2.metric("👥 Largest Segment", biggest_segment)
            c3.metric("🚨 At-Risk Customers", f"{at_risk_count:,}")

            st.markdown("---")
            st.markdown("#### 📝 Business Recommendations")
            st.markdown(
                "- **High-Value / Champion customers**: reward with loyalty programs & early access to new products.\n"
                "- **Regular / Loyal customers**: upsell and cross-sell using product recommendations.\n"
                "- **Occasional / Need Attention customers**: send re-engagement offers and personalized discounts.\n"
                "- **At-Risk / Hibernating customers**: win-back campaigns, surveys to understand drop-off reasons.\n"
                "- Use the **Product Recommendation** engine on the site/app to increase basket size.\n"
                "- Focus marketing budget on top revenue-generating countries identified in EDA."
            )

            st.markdown("#### Segment Distribution")
            st.dataframe(
                rfm_seg.groupby("Segment").agg(
                    Customers=("CustomerID", "count"),
                    Avg_Recency=("Recency", "mean"),
                    Avg_Frequency=("Frequency", "mean"),
                    Avg_Monetary=("Monetary", "mean"),
                    Total_Revenue=("Monetary", "sum"),
                ).round(1).sort_values("Total_Revenue", ascending=False),
                use_container_width=True,
            )