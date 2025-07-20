import pandas as pd
import numpy as np
import os
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Download necessary NLTK data (run once)
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment of a given text using VADER"""
    if isinstance(text, str):
        return sid.polarity_scores(text)
    else:
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}

def load_and_analyze(file_path):
    """Load a CSV file and analyze sentiment"""
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Apply sentiment analysis to title and selftext
    print(f"Analyzing sentiments in {file_path}...")
    
    # Create tqdm progress bar for sentiment analysis
    tqdm.pandas(desc="Analyzing title sentiment")
    df['title_sentiment'] = df['title'].progress_apply(lambda x: analyze_sentiment(x)['compound'])
    
    tqdm.pandas(desc="Analyzing selftext sentiment")
    df['selftext_sentiment'] = df['selftext'].progress_apply(lambda x: analyze_sentiment(x)['compound'])
    
    # Add combined sentiment (average of title and selftext)
    df['combined_sentiment'] = (df['title_sentiment'] + df['selftext_sentiment']) / 2
    
    # Categorize sentiment
    def categorize_sentiment(score):
        if score <= -0.05:
            return 'negative'
        elif score >= 0.05:
            return 'positive'
        else:
            return 'neutral'
    
    df['sentiment_category'] = df['combined_sentiment'].apply(categorize_sentiment)
    
    return df

def main():
    # Path to data directory
    data_dir = r'OriginalRedditData\rawData\2022'
    
    # Get all CSV files from the directory structure
    all_files = []
    for month_dir in os.listdir(data_dir):
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            csv_files = glob.glob(os.path.join(month_path, '*.csv'))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} CSV files to analyze")
    
    # Process each file and combine results
    all_data = []
    for file_path in all_files:
        try:
            df = load_and_analyze(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save processed data
        combined_df.to_csv('mental_health_sentiment_analysis.csv', index=False)
        
        # Visualize sentiment distribution by subreddit
        analyze_results(combined_df)
    else:
        print("No data was successfully processed.")

def analyze_results(df):
    # Other visualizations remain the same...
    
    # Average sentiment score by subreddit - IMPROVED VERSION
    plt.figure(figsize=(16, 10))
    
    # Group by subreddit and get the count and mean sentiment
    subreddit_stats = df.groupby('subreddit').agg({
        'combined_sentiment': 'mean',
        'author': 'count'  # Using 'author' column to count posts
    }).reset_index()
    
    # Rename columns for clarity
    subreddit_stats.columns = ['subreddit', 'avg_sentiment', 'post_count']
    
    # Filter to only include subreddits with a minimum number of posts (e.g., 100)
    min_posts = 100
    filtered_stats = subreddit_stats[subreddit_stats['post_count'] >= min_posts]
    
    # Sort by average sentiment
    filtered_stats = filtered_stats.sort_values('avg_sentiment')
    
    # Create a color map based on sentiment (red for negative, green for positive)
    colors = ['#d73027' if x < 0 else '#4575b4' for x in filtered_stats['avg_sentiment']]
    
    # Create the bar plot
    ax = sns.barplot(x='subreddit', y='avg_sentiment', data=filtered_stats, palette=colors)
    
    # Add post count as text on each bar
    for i, row in enumerate(filtered_stats.itertuples()):
        ax.text(i, row.avg_sentiment - 0.02, f"n={row.post_count}", 
                ha='center', va='top', color='white', fontsize=8)
    
    plt.title('Average Sentiment Score by Subreddit (Minimum 100 posts)')
    plt.xlabel('Subreddit')
    plt.ylabel('Average Sentiment Score')
    plt.axhline(y=0, color='r', linestyle='-')
    
    # Rotate labels and make them readable
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Adjust layout to make room for labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('avg_sentiment_by_subreddit.png', dpi=300)
    
    # For even better readability, create a horizontal bar chart with the top 20 subreddits
    plt.figure(figsize=(12, 14))
    
    # Get top 20 most active subreddits
    top_20 = subreddit_stats.sort_values('post_count', ascending=False).head(20)
    top_20 = top_20.sort_values('avg_sentiment')
    
    # Create horizontal bar chart
    colors = ['#d73027' if x < 0 else '#4575b4' for x in top_20['avg_sentiment']]
    ax = sns.barplot(x='avg_sentiment', y='subreddit', data=top_20, palette=colors)
    
    # Add count annotations
    for i, row in enumerate(top_20.itertuples()):
        ax.text(row.avg_sentiment - 0.02, i, f"n={row.post_count}", 
                ha='right' if row.avg_sentiment < 0 else 'left', 
                va='center', fontsize=9, color='black')
    
    plt.title('Average Sentiment Score by Top 20 Most Active Subreddits')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('top20_subreddits_sentiment.png', dpi=300)

if __name__ == "__main__":
    main()