import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from glob import glob
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize NLTK resources safely
def setup_nltk():
    try:
        import nltk
        # Download necessary NLTK resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error downloading {resource}: {e}")
        
        # Test if resources are available
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        stopwords.words('english')  # This will raise an error if stopwords aren't properly downloaded
        return True
    except Exception as e:
        print(f"NLTK setup failed: {e}")
        print("Will fall back to basic text processing methods")
        return False

# Text preprocessing with NLTK 
def preprocess_text(text, use_nltk=True):
    if not isinstance(text, str) or pd.isna(text) or text == '':
        return ''
    
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        if use_nltk:
            try:
                # Use NLTK for tokenization and stopword removal
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
                
                # Lemmatize
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            except Exception as e:
                print(f"NLTK processing failed: {e}")
                # Fallback to basic processing
                tokens = text.split()
                simple_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                                   "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                                   'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                                   'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                                   'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                                   'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                                   'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                                   'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                                   'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                                   'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                                   'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
                tokens = [word for word in tokens if word not in simple_stopwords]
        else:
            # Basic processing without NLTK
            tokens = text.split()
            simple_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                               "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                               'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                               'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                               'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                               'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                               'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                               'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                               'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with'}
            tokens = [word for word in tokens if word not in simple_stopwords]
        
        # Join tokens back to text
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return ''

# Sentiment analysis function that tries VADER if available, otherwise uses basic approach
def analyze_sentiment(text, use_nltk=True):
    if not isinstance(text, str) or pd.isna(text) or text == '':
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    
    if use_nltk:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)
        except Exception as e:
            print(f"VADER sentiment analysis failed: {e}")
            # Fall back to basic approach
    
    # Basic sentiment analysis as fallback
    # Mental health specific positive and negative word lists
    positive_words = {'good', 'great', 'happy', 'positive', 'excellent', 'wonderful', 'love',
                     'nice', 'amazing', 'awesome', 'fantastic', 'better', 'best', 'helpful',
                     'hope', 'improve', 'improving', 'improvement', 'support', 'supporting',
                     'supported', 'like', 'liked', 'well', 'benefit', 'benefits', 'benefited',
                     'success', 'successful', 'successfully', 'progress', 'progressive'}
    
    negative_words = {'bad', 'sad', 'negative', 'terrible', 'horrible', 'hate', 'awful',
                     'worse', 'worst', 'difficult', 'anxious', 'anxiety', 'depressed',
                     'depression', 'lonely', 'loneliness', 'stress', 'stressed', 'stressful',
                     'hopeless', 'suicidal', 'suicide', 'hurt', 'hurting', 'pain', 'painful',
                     'suffer', 'suffering', 'struggled', 'struggling', 'failure', 'fail',
                     'failed', 'failing', 'worry', 'worried', 'worrying', 'afraid', 'fear'}
    
    # Simple tokenization
    words = set(text.lower().split())
    
    # Count positive and negative words
    pos_count = len(words.intersection(positive_words))
    neg_count = len(words.intersection(negative_words))
    total_count = len(words)
    
    # Avoid division by zero
    if total_count == 0:
        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    
    # Calculate simple sentiment scores
    pos_score = pos_count / total_count
    neg_score = neg_count / total_count
    neu_score = 1 - (pos_score + neg_score)
    
    # Calculate compound score (simple difference between positive and negative)
    compound_score = (pos_score - neg_score) * 2  # Scale to be roughly between -1 and 1
    
    return {'neg': neg_score, 'neu': neu_score, 'pos': pos_score, 'compound': compound_score}

# Function to load mental health dataset
def load_mental_health_data():
    # Define possible paths
    possible_paths = ['raw data', 'raw_data', 'data', os.path.join('archive (1)', 'raw data'), '.']
    csv_files = []
    
    # Try to find CSV files
    for path in possible_paths:
        if os.path.exists(path):
            # Try direct search in this path
            found_files = glob(os.path.join(path, '/*.csv'), recursive=True)
            if found_files:
                csv_files = found_files
                print(f"Found {len(found_files)} CSV files in {path}")
                break
    
    # If no files found, try broader search
    if not csv_files:
        csv_files = glob('/*.csv', recursive=True)
        print(f"Found {len(csv_files)} CSV files in broader search")
    
    # If still no files, print available directories
    if not csv_files:
        print("No CSV files found. Available directories:", os.listdir())
        return pd.DataFrame()
    
    # Load and process files
    all_dataframes = []
    for file_path in csv_files:
        try:
            # Extract subreddit name from filename for metadata
            filename = os.path.basename(file_path)
            
            # Try to extract subreddit from the filename pattern
            if 'anx' in filename.lower():
                subreddit_from_filename = 'anxiety'
            elif 'dep' in filename.lower():
                subreddit_from_filename = 'depression'
            elif 'lone' in filename.lower():
                subreddit_from_filename = 'loneliness'
            elif 'mentalhealth' in filename.lower():
                subreddit_from_filename = 'mentalhealth'
            elif 'swap' in filename.lower() or 'suicide' in filename.lower():
                subreddit_from_filename = 'suicidewatch'
            else:
                subreddit_from_filename = filename.split('.')[0]
            
            # Load the CSV with error handling
            try:
                print(f"Loading {file_path}...")
                df = pd.read_csv(file_path)
                
                # Check if the dataframe already has a subreddit column
                if 'subreddit' not in df.columns:
                    df['subreddit'] = subreddit_from_filename
                
                # Add source information
                df['source_file'] = filename
                
                # Extract month from filename or path if needed
                if 'timestamp' in df.columns:
                    # Try to parse timestamp to get month
                    try:
                        df['month'] = pd.to_datetime(df['timestamp']).dt.strftime('%b').str.lower()
                    except:
                        df['month'] = 'unknown'
                else:
                    # Try to extract from filename
                    month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                    month_match = re.search(month_pattern, filename.lower())
                    if month_match:
                        df['month'] = month_match.group(1)
                    else:
                        df['month'] = 'unknown'
                
                all_dataframes.append(df)
                print(f"Loaded {filename} with {len(df)} rows")
                
                # Print sample columns to diagnose structure
                print(f"Columns in {filename}: {df.columns.tolist()[:5]}...")
                
            except UnicodeDecodeError:
                print(f"Trying different encoding for {filename}")
                df = pd.read_csv(file_path, encoding='latin1')
                
                if 'subreddit' not in df.columns:
                    df['subreddit'] = subreddit_from_filename
                
                df['source_file'] = filename
                all_dataframes.append(df)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_dataframes:
        print("No data loaded successfully")
        return pd.DataFrame()
    
    # Try to combine dataframes
    try:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Successfully combined {len(all_dataframes)} dataframes with {len(combined_df)} total rows")
        return combined_df
    except Exception as e:
        print(f"Error combining dataframes: {e}")
        print("Using first dataframe as fallback")
        return all_dataframes[0] if all_dataframes else pd.DataFrame()

# Main execution flow
try:
    print("Starting Reddit Mental Health Analysis...")
    
    # First, try to set up NLTK - use the result to determine if we can use NLTK features
    use_nltk = setup_nltk()
    
    # Load data
    df = load_mental_health_data()
    
    if df.empty:
        print("No data loaded, exiting analysis")
        exit()
    
    print(f"\nLoaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # The dataset has 'selftext' for post content and 'title' for post titles
    # Create combined text field for analysis
    print("\nPreparing text for analysis...")
    
    # Check if expected columns exist
    text_columns = []
    if 'selftext' in df.columns:
        text_columns.append('selftext')
    if 'title' in df.columns:
        text_columns.append('title')
    
    # If standard columns aren't found, try to find text columns
    if not text_columns:
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).str.len().mean()
                if sample > 20:  # Likely a text column if average length > 20 chars
                    text_columns.append(col)
                    print(f"Using {col} as text column (avg length: {sample:.1f})")
    
    if not text_columns:
        print("No text columns found for analysis")
        exit()
    
    # Create combined text
    df['combined_text'] = ''
    for col in text_columns:
        df['combined_text'] += ' ' + df[col].fillna('').astype(str)
    
    # Preprocess text
    print("Preprocessing text...")
    df['clean_text'] = df['combined_text'].apply(lambda x: preprocess_text(x, use_nltk))
    
    # Perform sentiment analysis
    print("Analyzing sentiment...")
    df['sentiment_scores'] = df['combined_text'].apply(lambda x: analyze_sentiment(x, use_nltk))
    df['negative'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['positive'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    
    # Categorize sentiment
    df['sentiment_category'] = df['compound'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_category', data=df)
    plt.title('Distribution of Sentiment in Mental Health Posts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png')
    plt.close()
    print("Created sentiment distribution visualization")
    
    # 2. Sentiment by subreddit
    if df['subreddit'].nunique() > 1:
        plt.figure(figsize=(12, 6))
        subreddit_sentiment = df.groupby('subreddit')['compound'].mean().sort_values()
        subreddit_sentiment.plot(kind='bar')
        plt.title('Average Sentiment by Mental Health Subreddit')
        plt.xlabel('Subreddit')
        plt.ylabel('Average Sentiment Score')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('subreddit_sentiment.png')
        plt.close()
        print("Created subreddit sentiment comparison")
    
    # 3. Word frequencies 
    try:
        print("\nAnalyzing word frequencies...")
        all_words = ' '.join(df['clean_text']).split()
        word_counts = Counter(all_words)
        
        # Get the top 20 words
        top_words = word_counts.most_common(20)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts)
        plt.title('Top 20 Words Across Mental Health Subreddits')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_words.png')
        plt.close()
        print("Created top words visualization")
        
        # 4. Word frequency by subreddit (if multiple subreddits)
        if df['subreddit'].nunique() > 1:
            plt.figure(figsize=(15, 10))
            for i, subreddit in enumerate(df['subreddit'].unique(), 1):
                if i <= 6:  # Limit to 6 subreddits for the plot
                    plt.subplot(2, 3, i)
                    subset = df[df['subreddit'] == subreddit]
                    if not subset.empty:
                        sub_words = ' '.join(subset['clean_text']).split()
                        sub_counts = Counter(sub_words).most_common(10)
                        if sub_counts:
                            sub_words, sub_counts = zip(*sub_counts)
                            plt.bar(sub_words, sub_counts)
                            plt.title(f'Top Words: r/{subreddit}')
                            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('subreddit_top_words.png')
            plt.close()
            print("Created subreddit word frequency comparison")
    except Exception as e:
        print(f"Error in word frequency analysis: {e}")
    
    # 5. Try to extract topics using LDA
    if len(df) >= 50:
        try:
            print("\nPerforming topic modeling...")
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=500)
            dtm = vectorizer.fit_transform(df['clean_text'])
            
            # Create LDA model
            num_topics = min(5, max(2, df['subreddit'].nunique()))
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(dtm)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Print top words for each topic
            print("\nTop words in each topic:")
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                print(f"Topic {topic_idx+1}: {', '.join(top_words)}")
            
            # Visualize topic distribution
            doc_topics = lda.transform(dtm)
            df['primary_topic'] = np.argmax(doc_topics, axis=1) + 1
            
            plt.figure(figsize=(10, 6))
            topic_counts = df['primary_topic'].value_counts().sort_index()
            plt.bar(topic_counts.index, topic_counts.values)
            plt.title('Document Distribution Across Topics')
            plt.xlabel('Topic')
            plt.ylabel('Number of Documents')
            plt.xticks(np.arange(1, num_topics+1))
            plt.savefig('topic_distribution.png')
            plt.close()
            print("Created topic model visualization")
        except Exception as e:
            print(f"Error in topic modeling: {e}")
    
    # Generate final report
    print("\n===== REDDIT MENTAL HEALTH ANALYSIS REPORT =====")
    print(f"Total posts analyzed: {len(df)}")
    
    print(f"\nMental health subreddits analyzed: {df['subreddit'].nunique()}")
    subreddit_counts = df['subreddit'].value_counts()
    for subreddit, count in subreddit_counts.items():
        print(f"  {subreddit}: {count} posts")
    
    print("\nSentiment Analysis:")
    sentiment_counts = df['sentiment_category'].value_counts()
    for category, count in sentiment_counts.items():
        print(f"  {category}: {count} posts ({count/len(df)*100:.1f}%)")
    
    if df['subreddit'].nunique() > 1:
        print("\nSentiment by Subreddit:")
        for subreddit in df['subreddit'].unique():
            avg_sentiment = df[df['subreddit'] == subreddit]['compound'].mean()
            print(f"  {subreddit}: {avg_sentiment:.4f} average sentiment")
    
    print("\nTop 10 Most Common Words:")
    for word, count in word_counts.most_common(10):
        print(f"  {word}: {count} occurrences")
    
    print("\nAnalysis complete! Visualizations saved as PNG files.")
    
except Exception as e:
    print(f"Critical error in analysis: {e}")
    import traceback
    traceback.print_exc()