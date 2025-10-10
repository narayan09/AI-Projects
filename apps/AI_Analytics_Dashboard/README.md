# 🎯 Jira Ticket Analytics Dashboard

A comprehensive Streamlit-based AI-powered dashboard for analyzing Jira ticket data with trend analysis, topic modeling, sentiment analysis, and predictive insights.

## 🚀 Features

### Core Analytics
- **📈 Trend Analysis**: Visualize ticket creation patterns over time, priority distributions, and assignee workloads
- **🏷️ AI-Powered Categorization**: Automatic topic modeling using BERTopic, SentenceTransformers, or keyword clustering
- **😊 Sentiment Analysis**: Analyze ticket sentiment using Hugging Face models or rule-based approaches
- **⏱️ Resolution Time Prediction**: ML-based prediction of ticket resolution times using Random Forest
- **🧠 AI Insights**: Automated insights generation based on data patterns

### Technical Highlights
- **Lightweight AI Models**: Uses efficient models like `all-MiniLM-L6-v2` for fast processing
- **Multiple AI Backends**: Supports HuggingFace Inference API (free tier), local models, and fallback methods
- **Real-time Processing**: Interactive dashboard with immediate feedback
- **Flexible Data Input**: Supports CSV and Excel formats with automatic column mapping

## 📋 Requirements

### System Requirements
- Python 3.9+
- 4GB+ RAM (for local AI models)
- Internet connection (for HuggingFace API)

### Dependencies
```bash
pip install -r requirements.txt
```

**Core Libraries:**
- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation and analysis
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning algorithms
- `sentence-transformers` - Text embeddings (optional)
- `requests` - API calls to HuggingFace

## 🔧 Installation & Setup

### 1. Clone/Download Project Files
```bash
# Download the project files:
# - app.py
# - utils.py
# - requirements.txt
# - sample_tickets.csv
# - README.md
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get HuggingFace API Token (Optional but Recommended)
1. Go to [HuggingFace.co](https://huggingface.co/)
2. Create a free account
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token for use in the dashboard

### 4. Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### Data Upload Options

#### Option 1: Use Sample Data
- Check "Use Sample Data" in the sidebar
- Includes 100 realistic Jira tickets with various categories

#### Option 2: Upload Your Jira Export
1. Export tickets from Jira as CSV or Excel
2. Ensure columns include: `Ticket ID`, `Summary`, `Description`, `Priority`, `Status`, `Assignee`, `Created Date`
3. Upload via the file uploader in the sidebar

### Dashboard Navigation

#### 📈 Trends Tab
- **Time Series**: Ticket creation trends over time
- **Priority Distribution**: Pie chart of priority levels
- **Status Overview**: Current ticket status distribution
- **Assignee Workload**: Top assignees by ticket count

#### 🏷️ Categories Tab
- **AI Topic Modeling**: Automatic categorization using AI
- **Topic Trends**: How categories change over time
- **Category Details**: Drill down into specific topics
- Sample tickets for each category

#### 😊 Sentiment Tab
- **Sentiment Distribution**: Overall positive/negative/neutral breakdown
- **Sentiment by Priority**: Correlation between sentiment and priority
- **Sample Analysis**: View sentiment scores for individual tickets

#### ⏱️ Predictions Tab
- **Resolution Time Prediction**: ML model predictions vs actual
- **Accuracy Metrics**: MAE and RMSE for model performance
- **Priority Analysis**: Average resolution time by priority level

#### 🧠 AI Insights Tab
- **Automated Insights**: AI-generated observations about your data
- **Trend Analysis**: Automatic detection of patterns and anomalies
- **Recommendations**: Suggested actions based on data analysis

### AI Configuration

#### HuggingFace API Token
- **Free Tier**: 30,000 characters/month, sufficient for small datasets
- **Benefits**: More accurate topic modeling and sentiment analysis
- **Fallback**: Works without token using rule-based methods

#### Performance Settings
- **Number of Topics**: Adjust for your dataset size (3-15 recommended)
- **Sample Limits**: Automatically limits processing for performance
- **Caching**: Results are cached for faster subsequent loads

## 🛠️ Customization

### Adding New Data Sources
Modify `load_and_process_data()` in `utils.py` to support additional formats:

```python
def load_and_process_data(file=None):
    if file.name.endswith('.json'):
        df = pd.read_json(file)
    # Add your custom logic here
    return process_jira_data(df)
```

### Custom Topic Categories
Edit the `keyword_based_clustering()` function to define your own categories:

```python
categories = {
    0: ['your', 'custom', 'keywords'],
    1: ['another', 'category', 'set'],
    # Add more categories
}
```

### New Visualizations
Add custom charts in `app.py`:

```python
# In the appropriate tab
fig_custom = px.scatter(df, x='column1', y='column2', title='Custom Chart')
st.plotly_chart(fig_custom, use_container_width=True)
```

## 🔍 Troubleshooting

### Common Issues

#### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

#### "HuggingFace API Error"
- Check your API token is valid
- Verify internet connection
- Free tier has usage limits - app will fallback to local methods

#### "Memory Error with Large Datasets"
- Use sample data or smaller datasets first
- Reduce `n_topics` parameter
- Consider upgrading hardware for large datasets

#### "Empty/Incorrect Visualizations"
- Check column names match expected format
- Ensure date columns are properly formatted
- Verify data contains the required fields

### Performance Optimization

#### For Slow Computers
1. Set `n_topics` to lower values (3-5)
2. Use sample data first
3. Disable local sentence transformers (rely on API only)
4. Process smaller chunks of data

#### For Large Datasets
1. Implement data sampling in `load_and_process_data()`
2. Use batch processing for AI features
3. Consider deploying to cloud platforms (Streamlit Cloud, AWS, etc.)

## 📁 Project Structure

```
jira-analytics-dashboard/
├── app.py                 # Main Streamlit application
├── utils.py              # AI/ML utility functions
├── requirements.txt      # Python dependencies
├── sample_tickets.csv    # Sample dataset
└── README.md            # This file
```

### File Descriptions

- **`app.py`**: Main dashboard interface with Streamlit components, tabs, and visualizations
- **`utils.py`**: Backend functions for data processing, AI models, and analysis
- **`requirements.txt`**: All required Python packages with versions
- **`sample_tickets.csv`**: 100 realistic sample tickets for testing
- **`README.md`**: Complete setup and usage documentation

## 🚀 Deployment

### Streamlit Cloud (Free)
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and deploy
4. Add HuggingFace token in secrets

### Local Network
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

### Adding New Features
1. Create new functions in `utils.py`
2. Add UI components in `app.py`
3. Update requirements if needed
4. Test with sample data

### Reporting Issues
- Include error messages
- Describe data format
- Mention system specifications
- Provide sample data if possible

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace**: For providing free AI model APIs
- **Streamlit**: For the excellent web framework
- **scikit-learn**: For machine learning algorithms
- **Plotly**: For interactive visualizations

---

## 🎯 Quick Start Summary

1. **Install**: `pip install -r requirements.txt`
2. **Run**: `streamlit run app.py`
3. **Upload**: Your Jira CSV/Excel or use sample data
4. **Analyze**: Explore trends, categories, sentiment, and predictions
5. **Insights**: Get AI-powered recommendations

**Need help?** Check the troubleshooting section or create an issue in the repository.