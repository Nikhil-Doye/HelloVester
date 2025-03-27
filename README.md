# Crypto RAG Analytics

An AI-powered cryptocurrency analysis platform combining **Retrieval-Augmented Generation (RAG)** with temporal pattern recognition for intelligent market insights.

## ✨ Features

- **Temporal Intelligence**: Analyze price/volume trends across custom date ranges
- **Performance Scoring**: Custom metric combining volatility and liquidity
- **AI-Powered Insights**: DeepSeek LLM integration for natural language analysis
- **Vector Search**: ChromaDB-backed semantic retrieval of historical data
- **Interactive Visualizations**: Plotly-powered charts with hover details

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [DeepSeek API Key](https://platform.deepseek.com/)
- Git

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/crypto-rag.git
cd crypto-rag

# Install dependencies
pip install -r requirements.txt

# Set up Streamlit secrets
mkdir -p .streamlit
echo "DEEPSEEK_API_KEY = 'your-api-key-here'" > .streamlit/secrets.toml
