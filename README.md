# Shopper Spectrum — Customer Analytics & Recommendation System

## About the Project

Shopper Spectrum is a Customer Analytics and Product Recommendation System built using
Python and Streamlit. It analyzes e-commerce transaction data to understand customer
shopping behaviour, segment customers into meaningful groups using RFM analysis and
KMeans clustering, and recommend products using similarity analysis.

## Objective

- Analyze customer shopping behaviour from transaction data.
- Segment customers into groups (High-Value, Loyal, At-Risk, etc.) using RFM analysis
  and machine learning.
- Recommend products to customers based on purchase similarity patterns.

## Business Problems 

- Who are the most valuable customers?
- Which customers buy regularly vs. rarely?
- Which customers have stopped shopping (at-risk)?
- Which products sell the most?
- What should be recommended to each customer?
- Which country generates the highest revenue?

## Tech Stack

- **Python**
- **Streamlit** — interactive web app framework
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive data visualization
- **Scikit-learn** — KMeans clustering, StandardScaler, cosine similarity

## Dataset

The app expects a CSV file with the following columns:

| Column | Description |
---
| InvoiceNo | Bill number |
| StockCode | Product code |
| Description | Product name |
| Quantity | Number of products purchased |
| InvoiceDate | Purchase date and time |
| UnitPrice | Price of one product |
| CustomerID | Customer number |
| Country | Customer's country |

## How to Run

1. Open a terminal in the project folder (where `app.py` is located).
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. The app will open in your browser. Upload your CSV file from the sidebar to get started.

## App Navigation / Features

- **Project Overview** — objective, business problems, and project workflow.
- **Dashboard** — key metrics (revenue, orders, customers, countries) and trend charts.
- **Data Cleaning** — handles missing values, duplicates, cancelled orders, and invalid
  entries, with a summary report of rows removed at each step.
- **EDA & Insights** — top-selling products, country-wise sales, monthly trends, and most
  active customers.
- **RFM Analysis** — calculates Recency, Frequency, and Monetary value for every customer.
- **Elbow Method** — determines the optimal number of clusters (K) for segmentation.
- **Customer Segmentation** — KMeans clustering with an adjustable number of clusters;
  segments are automatically labeled (Champion, High-Value, Loyal, Regular, Occasional,
  At-Risk) based on their RFM profile.
- **Product Similarity** — item-based collaborative filtering using cosine similarity to
  find products frequently bought together.
- **Product Recommendation** — select any product to get a list of top similar product
  recommendations.
- **Customer Prediction** — manually enter Recency, Frequency, and Monetary values for any
  customer to predict which segment they belong to.
- **Final Insights** — summary of top-performing segments, at-risk customer count, and
  actionable business recommendations.

## Important Note

Visit the **Customer Segmentation** page at least once before using the **Customer
Prediction** page, since the clustering model is trained there. All other pages work
independently once data is uploaded.

## 📌 Future Improvements

- Add customer lifetime value (CLV) prediction.
- Support multiple file formats (Excel, JSON).
- Add downloadable PDF/Excel reports for each analysis.
- Deploy on Streamlit Cloud for public access.
