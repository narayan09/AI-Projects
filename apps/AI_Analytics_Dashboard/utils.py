import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings("ignore")

# AI/ML Libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

def load_and_process_data(file=None):
    """
    Load and process Jira ticket data from file or create sample data
    """
    if file is None:
        # Create sample data
        return create_sample_data()
    
    try:
        # Read uploaded file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Process the data
        df = process_jira_data(df)
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_sample_data():
    """
    Create sample Jira ticket data for demonstration
    """
    np.random.seed(42)
    
    # Sample data parameters
    n_tickets = 500
    
    # Generate sample tickets
    ticket_ids = [f"PROJ-{i+1000}" for i in range(n_tickets)]
    
    # Sample summaries with different categories
    bug_summaries = [
        "Application crashes when user clicks submit button",
        "Login page not loading properly",
        "Database connection timeout error",
        "Page loads very slowly",
        "Error 500 when accessing user profile",
        "Cannot save changes to document",
        "Search function returns no results",
        "Image upload fails with large files",
        "Memory leak in background process",
        "API endpoint returning incorrect data"
    ]
    
    feature_summaries = [
        "Add dark mode to user interface",
        "Implement two-factor authentication",
        "Create export functionality for reports",
        "Add email notifications for updates",
        "Integrate with third-party calendar",
        "Implement user role management",
        "Add bulk operations for data management",
        "Create mobile-responsive design",
        "Add advanced search filters",
        "Implement audit logging system"
    ]
    
    support_summaries = [
        "User unable to reset password",
        "Help with configuring account settings",
        "Request for additional user permissions",
        "Question about billing and invoicing",
        "Need assistance with data migration",
        "Request for training documentation",
        "Account access issues after update",
        "Configuration help for new features",
        "User reporting duplicate notifications",
        "Request for account deletion"
    ]
    
    all_summaries = bug_summaries + feature_summaries + support_summaries
    
    # Generate descriptions
    descriptions = []
    for summary in all_summaries:
        if "crash" in summary.lower() or "error" in summary.lower():
            descriptions.append(f"User reported: {summary}. This is causing significant disruption to workflow. Steps to reproduce: 1) Navigate to page 2) Perform action 3) Error occurs. Expected: Normal operation.")
        elif "add" in summary.lower() or "implement" in summary.lower():
            descriptions.append(f"Feature request: {summary}. This would improve user experience and productivity. Acceptance criteria: 1) Feature is accessible 2) Works on all browsers 3) Proper documentation provided.")
        else:
            descriptions.append(f"Support request: {summary}. User needs assistance with this functionality. Priority should be based on user impact and business requirements.")
    
    # Generate random ticket data
    data = {
        'ticket_id': np.random.choice(ticket_ids, n_tickets, replace=False),
        'summary': np.random.choice(all_summaries, n_tickets),
        'description': np.random.choice(descriptions, n_tickets),
        'priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_tickets, p=[0.3, 0.4, 0.25, 0.05]),
        'status': np.random.choice(['Open', 'In Progress', 'Resolved', 'Closed', 'To Do'], n_tickets, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'assignee': np.random.choice(['John Smith', 'Sarah Johnson', 'Mike Davis', 'Lisa Wilson', 'David Brown', 'Emma Garcia'], n_tickets),
        'reporter': np.random.choice(['Customer A', 'Customer B', 'Internal User', 'QA Team', 'Product Manager'], n_tickets),
        'created_date': pd.date_range(start='2024-01-01', end='2024-10-10', periods=n_tickets),
    }
    
    df = pd.DataFrame(data)
    
    # Add resolved dates for resolved/closed tickets
    resolved_mask = df['status'].isin(['Resolved', 'Closed'])
    df.loc[resolved_mask, 'resolved_date'] = df.loc[resolved_mask, 'created_date'] + pd.to_timedelta(
        np.random.exponential(scale=5, size=resolved_mask.sum()), unit='D'
    )
    
    # Process the sample data
    df = process_jira_data(df)
    
    return df

def process_jira_data(df):
    """
    Clean and process Jira ticket data
    """
    # Standardize column names (handle different Jira export formats)
    column_mapping = {
        'key': 'ticket_id',
        'issue key': 'ticket_id',
        'issue_key': 'ticket_id',
        'summary': 'summary',
        'issue summary': 'summary',
        'description': 'description',
        'issue description': 'description',
        'priority': 'priority',
        'issue priority': 'priority',
        'status': 'status',
        'issue status': 'status',
        'assignee': 'assignee',
        'issue assignee': 'assignee',
        'reporter': 'reporter',
        'issue reporter': 'reporter',
        'created': 'created_date',
        'created date': 'created_date',
        'resolved': 'resolved_date',
        'resolved date': 'resolved_date'
    }
    
    # Rename columns to standard format
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns=column_mapping)
    
    # Convert dates
    date_columns = ['created_date', 'resolved_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate resolution days
    if 'created_date' in df.columns and 'resolved_date' in df.columns:
        df['resolution_days'] = (df['resolved_date'] - df['created_date']).dt.days
    
    # Fill missing values
    df['summary'] = df['summary'].fillna('No Summary')
    df['description'] = df['description'].fillna('No Description')
    df['priority'] = df['priority'].fillna('Medium')
    df['status'] = df['status'].fillna('Open')
    df['assignee'] = df['assignee'].fillna('Unassigned')
    
    # Create combined text for analysis
    df['combined_text'] = df['summary'].astype(str) + " " + df['description'].astype(str)
    
    return df

def get_trend_data(df, period='W'):
    """
    Generate trend data for visualization
    """
    if 'created_date' not in df.columns:
        return pd.DataFrame()
    
    trend_data = df.groupby(df['created_date'].dt.to_period(period)).size().reset_index(name='count')
    trend_data['date'] = trend_data['created_date'].dt.start_time
    
    return trend_data

def perform_topic_modeling(df, n_topics=8, hf_token=None):
    """
    Perform topic modeling using lightweight clustering
    """
    try:
        texts = df['combined_text'].fillna('').tolist()
        
        # Method 1: Try HuggingFace Inference API (lightweight)
        if hf_token and REQUESTS_AVAILABLE:
            try:
                embeddings = get_embeddings_hf_api(texts[:100], hf_token)  # Limit for free tier
                if embeddings is not None:
                    topics_df, topic_labels = cluster_embeddings(df[:100], embeddings, n_topics)
                    if topics_df is not None:
                        return topics_df, topic_labels
            except Exception as e:
                print(f"HF API failed: {str(e)}")
        
        # Method 2: Try local SentenceTransformers (if available)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use lightweight model
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(texts[:200])  # Limit for performance
                topics_df, topic_labels = cluster_embeddings(df[:200], embeddings, n_topics)
                if topics_df is not None:
                    return topics_df, topic_labels
            except Exception as e:
                print(f"Local embeddings failed: {str(e)}")
        
        # Method 3: Fallback to keyword-based clustering
        return keyword_based_clustering(df, n_topics)
        
    except Exception as e:
        print(f"Topic modeling failed: {str(e)}")
        return None, {}

def get_embeddings_hf_api(texts, token, model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get embeddings using HuggingFace Inference API
    """
    try:
        import requests
        
        API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Process in batches due to API limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, min(len(texts), 50), batch_size):  # Limit total requests
            batch_texts = texts[i:i+batch_size]
            
            response = requests.post(API_URL, headers=headers, json={
                "inputs": batch_texts,
                "options": {"wait_for_model": True}
            })
            
            if response.status_code == 200:
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)
            else:
                print(f"API request failed: {response.status_code}")
                return None
        
        return np.array(all_embeddings)
        
    except Exception as e:
        print(f"HF API embedding failed: {str(e)}")
        return None

def cluster_embeddings(df, embeddings, n_topics):
    """
    Cluster embeddings into topics using KMeans
    """
    try:
        if not SKLEARN_AVAILABLE:
            return None, {}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Add topic labels to dataframe
        topics_df = df.copy()
        topics_df['topic'] = cluster_labels
        
        # Generate topic labels based on most common words
        topic_labels = {}
        for topic_id in range(n_topics):
            topic_texts = topics_df[topics_df['topic'] == topic_id]['combined_text'].tolist()
            topic_keywords = extract_keywords_from_texts(topic_texts)
            topic_labels[topic_id] = f"{topic_keywords[:3]} (Topic {topic_id})"
        
        return topics_df, topic_labels
        
    except Exception as e:
        print(f"Clustering failed: {str(e)}")
        return None, {}

def keyword_based_clustering(df, n_topics):
    """
    Fallback keyword-based clustering when embeddings are not available
    """
    try:
        # Define keyword categories
        categories = {
            0: ['bug', 'error', 'crash', 'issue', 'problem', 'fail'],
            1: ['feature', 'enhancement', 'add', 'new', 'implement'],
            2: ['login', 'authentication', 'password', 'access', 'permission'],
            3: ['performance', 'slow', 'timeout', 'speed', 'load'],
            4: ['ui', 'interface', 'design', 'layout', 'visual'],
            5: ['database', 'data', 'sql', 'query', 'connection'],
            6: ['api', 'integration', 'service', 'endpoint', 'external'],
            7: ['support', 'help', 'question', 'how to', 'documentation']
        }
        
        # Categorize tickets based on keywords
        topics_df = df.copy()
        topics_df['topic'] = 0  # Default category
        
        for topic_id, keywords in categories.items():
            if topic_id >= n_topics:
                break
            
            mask = topics_df['combined_text'].str.contains('|'.join(keywords), case=False, na=False)
            topics_df.loc[mask, 'topic'] = topic_id
        
        # Generate topic labels
        topic_labels = {
            0: "Bug Reports & Issues",
            1: "Feature Requests",
            2: "Authentication & Access",
            3: "Performance Issues",
            4: "UI/UX Related",
            5: "Database & Data",
            6: "API & Integrations",
            7: "Support & Documentation"
        }
        
        # Keep only topics that exist
        topic_labels = {k: v for k, v in topic_labels.items() if k < n_topics}
        
        return topics_df, topic_labels
        
    except Exception as e:
        print(f"Keyword clustering failed: {str(e)}")
        return None, {}

def extract_keywords_from_texts(texts, top_n=3):
    """
    Extract top keywords from a list of texts
    """
    try:
        # Simple word frequency approach
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            words = [word for word in words if word not in stop_words and len(word) > 2]
            all_words.extend(words)
        
        # Count word frequency
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, count in top_words]
        
    except:
        return ["keyword", "analysis", "failed"]

def categorize_tickets(df):
    """
    Basic ticket categorization without AI
    """
    try:
        categories_df = df.copy()
        
        # Simple rule-based categorization
        def categorize_ticket(text):
            text = text.lower()
            if any(word in text for word in ['bug', 'error', 'crash', 'issue', 'problem']):
                return 'Bug'
            elif any(word in text for word in ['feature', 'enhancement', 'add', 'new']):
                return 'Feature Request'
            elif any(word in text for word in ['support', 'help', 'question', 'how']):
                return 'Support'
            else:
                return 'Other'
        
        categories_df['category'] = categories_df['combined_text'].apply(categorize_ticket)
        return categories_df
        
    except Exception as e:
        print(f"Categorization failed: {str(e)}")
        return None

def analyze_sentiment(df, hf_token=None):
    """
    Analyze sentiment of ticket descriptions
    """
    try:
        # Try HuggingFace API first (if token provided)
        if hf_token and REQUESTS_AVAILABLE:
            try:
                texts = df['combined_text'].fillna('').tolist()[:50]  # Limit for API
                sentiments = get_sentiment_hf_api(texts, hf_token)
                
                if sentiments:
                    sentiment_df = df[:len(sentiments)].copy()
                    sentiment_df['sentiment'] = [s['label'].lower() for s in sentiments]
                    sentiment_df['sentiment_score'] = [s['score'] for s in sentiments]
                    return sentiment_df
                    
            except Exception as e:
                print(f"HF sentiment API failed: {str(e)}")
        
        # Fallback to rule-based sentiment
        return rule_based_sentiment(df)
        
    except Exception as e:
        print(f"Sentiment analysis failed: {str(e)}")
        return None

def get_sentiment_hf_api(texts, token, model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Get sentiment using HuggingFace Inference API
    """
    try:
        import requests
        
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Process in batches
        batch_size = 5
        all_sentiments = []
        
        for i in range(0, min(len(texts), 25), batch_size):  # Limit requests
            batch_texts = texts[i:i+batch_size]
            
            response = requests.post(API_URL, headers=headers, json={
                "inputs": batch_texts,
                "options": {"wait_for_model": True}
            })
            
            if response.status_code == 200:
                batch_results = response.json()
                # Handle both single and batch responses
                if isinstance(batch_results[0], list):
                    for result in batch_results:
                        all_sentiments.append(result[0])  # Take highest score
                else:
                    all_sentiments.extend(batch_results)
            else:
                print(f"Sentiment API failed: {response.status_code}")
                return None
        
        return all_sentiments
        
    except Exception as e:
        print(f"Sentiment API error: {str(e)}")
        return None

def rule_based_sentiment(df):
    """
    Simple rule-based sentiment analysis
    """
    try:
        sentiment_df = df.copy()
        
        def analyze_text_sentiment(text):
            text = text.lower()
            
            positive_words = ['good', 'great', 'excellent', 'working', 'solved', 'fixed', 'success', 'thank']
            negative_words = ['bad', 'terrible', 'broken', 'error', 'fail', 'crash', 'issue', 'problem', 'bug']
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                return 'positive', 0.7
            elif neg_count > pos_count:
                return 'negative', 0.7
            else:
                return 'neutral', 0.5
        
        sentiments = sentiment_df['combined_text'].apply(analyze_text_sentiment)
        sentiment_df['sentiment'] = [s[0] for s in sentiments]
        sentiment_df['sentiment_score'] = [s[1] for s in sentiments]
        
        return sentiment_df
        
    except Exception as e:
        print(f"Rule-based sentiment failed: {str(e)}")
        return None

def predict_resolution_time(df):
    """
    Predict resolution time using simple machine learning
    """
    try:
        if not SKLEARN_AVAILABLE:
            return None
        
        # Filter resolved tickets for training
        resolved_df = df[df['resolution_days'].notna()].copy()
        
        if len(resolved_df) < 10:
            return None
        
        # Feature engineering
        le_priority = LabelEncoder()
        le_assignee = LabelEncoder()
        
        resolved_df['priority_encoded'] = le_priority.fit_transform(resolved_df['priority'])
        resolved_df['assignee_encoded'] = le_assignee.fit_transform(resolved_df['assignee'])
        resolved_df['text_length'] = resolved_df['combined_text'].str.len()
        resolved_df['is_weekend'] = resolved_df['created_date'].dt.weekday >= 5
        
        # Features for model
        feature_cols = ['priority_encoded', 'assignee_encoded', 'text_length', 'is_weekend']
        X = resolved_df[feature_cols]
        y = resolved_df['resolution_days']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Create results dataframe
        results_df = resolved_df.iloc[X_test.index].copy()
        results_df['predicted_days'] = y_pred
        results_df['actual_days'] = y_test.values
        
        return results_df
        
    except Exception as e:
        print(f"Prediction modeling failed: {str(e)}")
        return None

def generate_insights(df, hf_token=None):
    """
    Generate AI insights about the ticket data
    """
    try:
        insights = []
        
        # Basic statistical insights
        total_tickets = len(df)
        
        if 'priority' in df.columns:
            high_priority_count = len(df[df['priority'] == 'High'])
            high_priority_pct = (high_priority_count / total_tickets) * 100
            
            if high_priority_pct > 20:
                insights.append(f"High priority tickets represent {high_priority_pct:.1f}% of all tickets, which may indicate systemic issues requiring attention.")
        
        if 'status' in df.columns:
            open_tickets = len(df[df['status'].isin(['Open', 'To Do', 'In Progress'])])
            open_pct = (open_tickets / total_tickets) * 100
            
            if open_pct > 60:
                insights.append(f"{open_pct:.1f}% of tickets are still open, suggesting potential resource constraints or process bottlenecks.")
        
        if 'resolution_days' in df.columns:
            avg_resolution = df['resolution_days'].mean()
            if avg_resolution > 10:
                insights.append(f"Average resolution time is {avg_resolution:.1f} days, which may be impacting customer satisfaction.")
        
        if 'assignee' in df.columns:
            assignee_counts = df['assignee'].value_counts()
            if len(assignee_counts) > 0:
                top_assignee = assignee_counts.index[0]
                top_count = assignee_counts.iloc[0]
                workload_pct = (top_count / total_tickets) * 100
                
                if workload_pct > 40:
                    insights.append(f"Workload appears unbalanced: {top_assignee} handles {workload_pct:.1f}% of all tickets.")
        
        # Trend insights
        if 'created_date' in df.columns:
            recent_week = df[df['created_date'] >= df['created_date'].max() - pd.Timedelta(days=7)]
            if len(recent_week) > len(df) * 0.2:
                insights.append("Ticket creation has significantly increased in the past week, indicating possible emerging issues.")
        
        return insights[:5]  # Return top 5 insights
        
    except Exception as e:
        print(f"Insight generation failed: {str(e)}")
        return []