# crypto_rag.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import chromadb
import plotly.express as px
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Streamlit configuration
st.set_page_config(
    page_title="Crypto Temporal Analyst",
    page_icon="📅",
    layout="wide"
)

# Initialize components
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# Load and validate data
try:
    df = pd.read_csv("crypto_data.csv", parse_dates=['Date'], dayfirst=False)
    df = df[df['Date'].between('2020-01-01', '2025-12-31')]
    df['Change'] = pd.to_numeric(df['Change'], errors='coerce').abs()
    df['PerformanceScore'] = (df['Change'] * df['Volume']) / 1e9
    df['timestamp'] = df['Date'].apply(lambda x: int(x.timestamp()))  # Add timestamp
except Exception as e:
    st.error(f"Data error: {str(e)}")
    st.stop()

# ChromaDB setup with explicit configuration
try:
    client = chromadb.PersistentClient(
        path="vector_store",
        tenant="default_tenant",
        database="default_database"
    )
    collection = client.get_or_create_collection(
        name="crypto_temporal",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    st.error(f"DB connection failed: {str(e)}")
    st.stop()

# Embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

model = load_model()

def get_date_range():
    min_date = df['Date'].min().strftime('%Y-%m-%d')
    max_date = df['Date'].max().strftime('%Y-%m-%d')
    return min_date, max_date

def validate_query_dates(query_dates):
    min_date, max_date = get_date_range()
    for q_date in query_dates:
        if not (datetime.strptime(min_date, "%Y-%m-%d") <= q_date <= datetime.strptime(max_date, "%Y-%m-%d")):
            st.warning(f"Date {q_date.date()} outside dataset range {min_date} to {max_date}")
            return False
    return True

def store_embeddings():
    if collection.count() == 0:
        batch_size = 5000
        total_rows = len(df)
        
        try:
            for batch_start in range(0, total_rows, batch_size):
                batch = df.iloc[batch_start:batch_start+batch_size]
                documents = []
                metadatas = []
                ids = []
                
                for idx, row in batch.iterrows():
                    documents.append(
                        f"{row['Coin Name']} | {row['Date'].date()} | "
                        f"Δ{row['Change']}% | ${row['Volume']/1e9:.1f}B"
                    )
                    metadatas.append({
                        "coin": row['Coin Name'],
                        "date_str": row['Date'].strftime("%Y-%m-%d"),
                        "timestamp": row['timestamp'],
                        "price": float(row['Price']),
                        "change": float(row['Change']),
                        "volume": float(row['Volume']),
                        "performance": float(row['PerformanceScore'])
                    })
                    ids.append(str(idx))
                
                embeddings = model.encode(documents, show_progress_bar=False)
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
        except Exception as e:
            st.error(f"Storage error: {str(e)}")
            st.stop()

def extract_query_dates(query):
    date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    raw_dates = re.findall(date_pattern, query)
    dates = []
    
    for date_str in raw_dates:
        try:
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            continue
            
    return dates

def retrieve_context(query):
    query_dates = extract_query_dates(query)
    
    # Date validation
    if query_dates:
        if not validate_query_dates(query_dates):
            return None
        start_ts = int(min(query_dates).timestamp())
        end_ts = int(max(query_dates).timestamp())
        where_clause = {
            "$and": [
                {"timestamp": {"$gte": start_ts}},
                {"timestamp": {"$lte": end_ts}}
            ]
        }
    else:
        where_clause = {}
    
    try:
        enhanced_query = f"{query} temporal analysis date range performance"
        query_embedding = model.encode(enhanced_query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=15,
            include=["metadatas"],
            where=where_clause
        )
        
        if not results['metadatas'][0]:
            st.warning("No data found for specified timeframe")
            return None
            
        sorted_results = sorted(
            results['metadatas'][0],
            key=lambda x: x['performance'],
            reverse=True
        )
        
        return {'metadatas': [sorted_results]}
        
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return None

def generate_rag_response(query, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    context_data = pd.DataFrame(context['metadatas'][0])
    date_range = ""
    if not context_data.empty:
        min_date = context_data['date_str'].min()
        max_date = context_data['date_str'].max()
        date_range = f"({min_date} to {max_date})"
    
    prompt = f"""Analyze cryptocurrency performance {date_range}:
    1. Identify top performers by price change and volume
    2. Compare daily/weekly trends
    3. Highlight significant market movements
    4. Provide numerical insights with exact values
    
    Context Data:
    {context_data.to_markdown(index=False)}
    
    Query: {query}
    Comprehensive analysis:"""
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You're a crypto temporal analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 1000
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Analysis error: {str(e)}"

def create_temporal_chart(context):
    df_context = pd.DataFrame(context['metadatas'][0])
    if not df_context.empty:
        df_context['date'] = pd.to_datetime(df_context['date_str'])
        fig = px.scatter(
            df_context,
            x='date', y='performance',
            color='coin',
            size='volume',
            hover_data=['price', 'change'],
            title="Temporal Performance Analysis",
            labels={'performance': 'Performance Score', 'date': 'Date'}
        )
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# Initialize
store_embeddings()

# UI
st.title("⏳ Crypto Temporal Analysis")
min_date, max_date = get_date_range()
st.caption(f"Analyzing data from {min_date} to {max_date}")

query = st.text_input("Enter temporal analysis query (e.g., 'Top performers Jan 2024'):")

if query:
    context = retrieve_context(query)
    
    if context and context['metadatas'][0]:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Temporal Analysis")
            response = generate_rag_response(query, context)
            st.markdown(response)
            
        with col2:
            st.subheader("Visual Timeline")
            create_temporal_chart(context)
    else:
        st.error("Could not retrieve data for this temporal query")

st.sidebar.markdown(f"""
**How to Query**:
- Use specific date ranges: "2024-01-01 to 2024-01-31"
- Ask about trends: "Weekly volume leaders March 2025"
- Compare periods: "BTC vs ETH performance Q1 2024"

**Data Features**:
- Price Change Tracking
- Volume Analysis
- Performance Scoring
- Temporal Correlation
""")