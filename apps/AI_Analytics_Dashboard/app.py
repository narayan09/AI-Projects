import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import utility functions
from utils import (
    load_and_process_data,
    get_trend_data,
    perform_topic_modeling,
    categorize_tickets,
    analyze_sentiment,
    predict_resolution_time,
    generate_insights
)

# Configure Streamlit page
st.set_page_config(
    page_title="Jira Ticket Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎯 Jira Ticket Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload and filters
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Jira Export (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your Jira ticket export in CSV or Excel format"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use Sample Data", value=not uploaded_file)
        
        if uploaded_file or use_sample:
            # Load and process data
            if uploaded_file:
                df = load_and_process_data(uploaded_file)
            else:
                df = load_and_process_data()  # Load sample data
            
            if df is not None and not df.empty:
                st.success(f"✅ Loaded {len(df)} tickets")
                
                # Date range filter
                st.header("🗓️ Filters")
                
                if 'created_date' in df.columns:
                    date_range = st.date_input(
                        "Select Date Range",
                        value=[df['created_date'].min(), df['created_date'].max()],
                        min_value=df['created_date'].min(),
                        max_value=df['created_date'].max()
                    )
                    
                    if len(date_range) == 2:
                        df = df[
                            (df['created_date'] >= pd.to_datetime(date_range[0])) &
                            (df['created_date'] <= pd.to_datetime(date_range[1]))
                        ]
                
                # Priority filter
                if 'priority' in df.columns:
                    priorities = st.multiselect(
                        "Select Priorities",
                        options=df['priority'].unique(),
                        default=df['priority'].unique()
                    )
                    df = df[df['priority'].isin(priorities)]
                
                # Status filter
                if 'status' in df.columns:
                    statuses = st.multiselect(
                        "Select Statuses",
                        options=df['status'].unique(),
                        default=df['status'].unique()
                    )
                    df = df[df['status'].isin(statuses)]
                
                st.header("🤖 AI Settings")
                
                # HuggingFace API Token
                hf_token = st.text_input(
                    "HuggingFace API Token (Optional)",
                    type="password",
                    help="Enter your HF token for better AI features. Free tier available!"
                )
                
                # Topic modeling settings
                n_topics = st.slider("Number of Topics", 3, 15, 8)
                use_ai_insights = st.checkbox("Generate AI Insights", value=True)
                
            else:
                st.error("❌ Failed to load data")
                return
        else:
            st.info("👆 Please upload a Jira export file or use sample data")
            return
    
    # Main dashboard content
    if df is not None and not df.empty:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Tickets</p></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            open_tickets = len(df[df['status'].isin(['Open', 'In Progress', 'To Do'])])
            st.markdown(
                f'<div class="metric-card"><h3>{open_tickets}</h3><p>Open Tickets</p></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            if 'priority' in df.columns:
                high_priority = len(df[df['priority'] == 'High'])
                st.markdown(
                    f'<div class="metric-card"><h3>{high_priority}</h3><p>High Priority</p></div>',
                    unsafe_allow_html=True
                )
        
        with col4:
            if 'resolution_days' in df.columns:
                avg_resolution = df['resolution_days'].mean()
                st.markdown(
                    f'<div class="metric-card"><h3>{avg_resolution:.1f}</h3><p>Avg Resolution (days)</p></div>',
                    unsafe_allow_html=True
                )
        
        with col5:
            if 'created_date' in df.columns:
                this_month = len(df[df['created_date'] >= pd.Timestamp.now() - pd.Timedelta(days=30)])
                st.markdown(
                    f'<div class="metric-card"><h3>{this_month}</h3><p>This Month</p></div>',
                    unsafe_allow_html=True
                )
        
        # Tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Trends", "🏷️ Categories", "😊 Sentiment", "⏱️ Predictions", "🧠 AI Insights"
        ])
        
        with tab1:
            st.subheader("Ticket Trends Analysis")
            
            # Time series plot
            if 'created_date' in df.columns:
                trend_data = get_trend_data(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_trend = px.line(
                        trend_data,
                        x='date',
                        y='count',
                        title='Tickets Created Over Time',
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_trend.update_layout(height=400)
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    # Priority distribution
                    if 'priority' in df.columns:
                        priority_counts = df['priority'].value_counts()
                        fig_priority = px.pie(
                            values=priority_counts.values,
                            names=priority_counts.index,
                            title='Priority Distribution'
                        )
                        fig_priority.update_layout(height=400)
                        st.plotly_chart(fig_priority, use_container_width=True)
            
            # Status distribution
            col1, col2 = st.columns(2)
            
            with col1:
                if 'status' in df.columns:
                    status_counts = df['status'].value_counts()
                    fig_status = px.bar(
                        x=status_counts.values,
                        y=status_counts.index,
                        orientation='h',
                        title='Status Distribution',
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_status.update_layout(height=400)
                    st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                if 'assignee' in df.columns:
                    assignee_counts = df['assignee'].value_counts().head(10)
                    fig_assignee = px.bar(
                        x=assignee_counts.index,
                        y=assignee_counts.values,
                        title='Top 10 Assignees',
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig_assignee.update_xaxes(tickangle=45)
                    fig_assignee.update_layout(height=400)
                    st.plotly_chart(fig_assignee, use_container_width=True)
        
        with tab2:
            st.subheader("AI-Powered Ticket Categorization")
            
            # Perform topic modeling
            with st.spinner("🤖 Analyzing ticket content with AI..."):
                try:
                    topics_df, topic_labels = perform_topic_modeling(df, n_topics=n_topics, hf_token=hf_token)
                    
                    if topics_df is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Topic distribution
                            topic_counts = topics_df['topic'].value_counts()
                            fig_topics = px.bar(
                                x=topic_counts.values,
                                y=[topic_labels.get(i, f"Topic {i}") for i in topic_counts.index],
                                orientation='h',
                                title='Ticket Categories (AI-Generated)',
                                color_discrete_sequence=['#9467bd']
                            )
                            fig_topics.update_layout(height=500)
                            st.plotly_chart(fig_topics, use_container_width=True)
                        
                        with col2:
                            # Topic over time
                            if 'created_date' in topics_df.columns:
                                topic_time = topics_df.groupby([
                                    topics_df['created_date'].dt.to_period('M'), 'topic'
                                ]).size().reset_index(name='count')
                                topic_time['period'] = topic_time['created_date'].astype(str)
                                
                                fig_topic_time = px.line(
                                    topic_time,
                                    x='period',
                                    y='count',
                                    color='topic',
                                    title='Topic Trends Over Time',
                                    labels={'topic': 'Category'}
                                )
                                fig_topic_time.update_layout(height=500)
                                st.plotly_chart(fig_topic_time, use_container_width=True)
                        
                        # Topic details
                        st.subheader("Category Details")
                        
                        selected_topic = st.selectbox(
                            "Select a category to explore:",
                            options=topic_labels.keys(),
                            format_func=lambda x: topic_labels.get(x, f"Topic {x}")
                        )
                        
                        if selected_topic is not None:
                            topic_tickets = topics_df[topics_df['topic'] == selected_topic]
                            st.write(f"**{topic_labels.get(selected_topic)}** - {len(topic_tickets)} tickets")
                            
                            # Show sample tickets
                            sample_tickets = topic_tickets[['ticket_id', 'summary', 'created_date']].head(5)
                            st.dataframe(sample_tickets, use_container_width=True)
                    
                    else:
                        st.warning("⚠️ Could not perform topic modeling. Using basic categorization.")
                        categories_df = categorize_tickets(df)
                        
                        if categories_df is not None:
                            category_counts = categories_df['category'].value_counts()
                            fig_cat = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title='Basic Categories'
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Topic modeling failed: {str(e)}")
                    st.info("💡 Try using the HuggingFace API token for better results")
        
        with tab3:
            st.subheader("Sentiment Analysis")
            
            with st.spinner("🎭 Analyzing sentiment..."):
                sentiment_df = analyze_sentiment(df, hf_token=hf_token)
                
                if sentiment_df is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment_counts = sentiment_df['sentiment'].value_counts()
                        fig_sentiment = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title='Overall Sentiment Distribution',
                            color_discrete_map={
                                'positive': '#2ca02c',
                                'negative': '#d62728',
                                'neutral': '#ff7f0e'
                            }
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    with col2:
                        # Sentiment by priority
                        if 'priority' in sentiment_df.columns:
                            sentiment_priority = pd.crosstab(
                                sentiment_df['priority'], 
                                sentiment_df['sentiment'], 
                                normalize='index'
                            ) * 100
                            
                            fig_sent_priority = px.bar(
                                sentiment_priority,
                                title='Sentiment by Priority (%)',
                                color_discrete_map={
                                    'positive': '#2ca02c',
                                    'negative': '#d62728',
                                    'neutral': '#ff7f0e'
                                }
                            )
                            st.plotly_chart(fig_sent_priority, use_container_width=True)
                    
                    # Sentiment examples
                    st.subheader("Sample Analysis")
                    
                    sentiment_filter = st.selectbox(
                        "Filter by sentiment:",
                        ['All'] + list(sentiment_df['sentiment'].unique())
                    )
                    
                    if sentiment_filter != 'All':
                        filtered_df = sentiment_df[sentiment_df['sentiment'] == sentiment_filter]
                    else:
                        filtered_df = sentiment_df
                    
                    sample_sentiment = filtered_df[['ticket_id', 'summary', 'sentiment', 'sentiment_score']].head(10)
                    st.dataframe(sample_sentiment, use_container_width=True)
                
                else:
                    st.warning("⚠️ Sentiment analysis not available without HuggingFace token")
        
        with tab4:
            st.subheader("Resolution Time Predictions")
            
            with st.spinner("🔮 Training prediction model..."):
                predictions_df = predict_resolution_time(df)
                
                if predictions_df is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Actual vs Predicted resolution times
                        fig_pred = px.scatter(
                            predictions_df,
                            x='actual_days',
                            y='predicted_days',
                            title='Actual vs Predicted Resolution Time',
                            labels={
                                'actual_days': 'Actual Days',
                                'predicted_days': 'Predicted Days'
                            }
                        )
                        # Add diagonal line for perfect prediction
                        fig_pred.add_trace(
                            go.Scatter(
                                x=[0, predictions_df['actual_days'].max()],
                                y=[0, predictions_df['actual_days'].max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(dash='dash', color='red')
                            )
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        # Resolution time by priority
                        if 'priority' in predictions_df.columns:
                            resolution_by_priority = predictions_df.groupby('priority')['predicted_days'].mean().sort_values()
                            
                            fig_res_priority = px.bar(
                                x=resolution_by_priority.values,
                                y=resolution_by_priority.index,
                                orientation='h',
                                title='Avg Predicted Resolution by Priority',
                                color_discrete_sequence=['#17becf']
                            )
                            st.plotly_chart(fig_res_priority, use_container_width=True)
                    
                    # Prediction accuracy metrics
                    if 'actual_days' in predictions_df.columns and 'predicted_days' in predictions_df.columns:
                        mae = np.mean(np.abs(predictions_df['actual_days'] - predictions_df['predicted_days']))
                        rmse = np.sqrt(np.mean((predictions_df['actual_days'] - predictions_df['predicted_days']) ** 2))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Absolute Error", f"{mae:.2f} days")
                        with col2:
                            st.metric("Root Mean Square Error", f"{rmse:.2f} days")
                
                else:
                    st.warning("⚠️ Not enough resolved tickets for prediction modeling")
        
        with tab5:
            st.subheader("AI-Generated Insights")
            
            if use_ai_insights and hf_token:
                with st.spinner("🧠 Generating insights..."):
                    insights = generate_insights(df, hf_token=hf_token)
                    
                    if insights:
                        for i, insight in enumerate(insights, 1):
                            st.markdown(
                                f'<div class="insight-box"><strong>Insight #{i}:</strong> {insight}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("💡 Enable AI insights and provide HuggingFace token for automated analysis")
            
            else:
                # Manual insights based on data analysis
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write("**Manual Data Insights:**")
                
                insights_list = []
                
                # Trend insights
                if 'created_date' in df.columns:
                    recent_tickets = len(df[df['created_date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)])
                    previous_tickets = len(df[
                        (df['created_date'] >= pd.Timestamp.now() - pd.Timedelta(days=14)) &
                        (df['created_date'] < pd.Timestamp.now() - pd.Timedelta(days=7))
                    ])
                    
                    if previous_tickets > 0:
                        change_pct = ((recent_tickets - previous_tickets) / previous_tickets) * 100
                        trend = "increased" if change_pct > 0 else "decreased"
                        insights_list.append(f"Ticket creation has {trend} by {abs(change_pct):.1f}% this week")
                
                # Priority insights
                if 'priority' in df.columns:
                    high_priority_pct = (len(df[df['priority'] == 'High']) / len(df)) * 100
                    if high_priority_pct > 30:
                        insights_list.append(f"High priority tickets make up {high_priority_pct:.1f}% of all tickets - consider reviewing prioritization")
                
                # Assignee insights
                if 'assignee' in df.columns:
                    top_assignee = df['assignee'].value_counts().index[0]
                    top_assignee_count = df['assignee'].value_counts().iloc[0]
                    total_assignees = df['assignee'].nunique()
                    insights_list.append(f"{top_assignee} handles {top_assignee_count} tickets ({(top_assignee_count/len(df)*100):.1f}% of total)")
                
                # Resolution insights
                if 'resolution_days' in df.columns:
                    avg_resolution = df['resolution_days'].mean()
                    if avg_resolution > 7:
                        insights_list.append(f"Average resolution time is {avg_resolution:.1f} days - consider process improvements")
                
                for insight in insights_list:
                    st.write(f"• {insight}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Raw data view (collapsible)
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
            
            # Download processed data
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv_data,
                file_name=f"processed_jira_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()