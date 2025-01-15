import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from textblob import TextBlob  # For sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer
#from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis 
import pyLDAvis.lda_model
import streamlit.components.v1 as components
from nltk.corpus import stopwords

#Download necessary package for NLP task
nltk.download('vader_lexicon')
# Download English language model for spaCy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# Load English NLP model
nlp = spacy.load('en_core_web_sm')
custom_stopwords = ['good', 'product', 'amazon', 'price', 'images', '_sy88', 'jpg', 'nice', 'quality', 'com']
# NLTK’s English stopwords come as a list, so convert them to a set
default_stopwords = set(stopwords.words('english'))
# Combine your custom stopwords with NLTK’s stopwords
all_stopwords_set = default_stopwords.union(custom_stopwords)
# Convert the set to a list
all_stopwords = list(all_stopwords_set)


# Load data  
df = pd.read_csv("C:/Users/Samuel Lin/Downloads/ML-Powered Web App/NLP/amazon.csv")
st.set_page_config(
    page_title="Amazon Customer Review Analysis Dashboard",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

def preprocess_data(data):
    """
    Clean and restructure Amazon product data.

    Args:
        data (pd.DataFrame):
            The input DataFrame with the following columns:
            - 'discounted_price'
            - 'actual_price'
            - 'discount_percentage'
            - 'rating'
            - 'rating_count'
            - 'category'
            - 'product_id'
            - 'product_name'
            - 'review_id'
            - 'review_content'
            (plus any additional columns used in the function logic).

    Returns:
        pd.DataFrame:
            A cleaned and restructured version of the original data with:
            - Numeric prices and rating columns
            - Split 'category' columns ('category_1', 'category_2', etc.)
            - A 'rating_score' categorical column
    """

    #Changing the data type of discounted price and actual price
    df['discounted_price'] = df['discounted_price'].str.replace("₹",'')
    df['discounted_price'] = df['discounted_price'].str.replace(",",'')
    df['discounted_price'] = df['discounted_price'].astype('float64')
    df['actual_price'] = df['actual_price'].str.replace("₹",'')
    df['actual_price'] = df['actual_price'].str.replace(",",'')
    df['actual_price'] = df['actual_price'].astype('float64')

    #Changing Datatype and values in Discount Percentage
    df['discount_percentage'] = df['discount_percentage'].str.replace('%','').astype('float64')
    df['discount_percentage'] = df['discount_percentage'] / 100

    #Changing Rating Columns Data Type (There is a record with rating '|')
    #I went to the amazon page to get the rating and found that the product id of B08L12N5H1 has a rating of 4. So I am going to give the item rating a 4.0 as well.
    # Source: https://www.amazon.in/Eureka-Forbes-Vacuum-Cleaner-Washable/dp/B08L12N5H1
    df['rating'] = df['rating'].str.replace('|', '4.0').astype('float64')
    #Changing Rating Column Data Type
    df['rating_count'] = df['rating_count'].str.replace(',', '').astype('float64')

    #Creating a new DataFrame with Selected Column
    df1 = df[['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'review_id', 'review_content']].copy()
    
    #Splitting the Strings in the category column
    catsplit = df['category'].str.split('|', expand=True)

    #Renaming category column
    catsplit = catsplit.rename(columns={0:'category_1', 1:'category_2', 2:'category_3'})
    #Adding categories to the new dataframe
    df1['category_1'] = catsplit['category_1']
    df1['category_2'] = catsplit['category_2']
    df1.drop(columns='category', inplace=True)

    #Fixing Strings in the Category_1 Column
    df1['category_1'] = df1['category_1'].str.replace('&', ' & ')
    df1['category_1'] = df1['category_1'].str.replace('OfficeProducts', 'Office Products')
    df1['category_1'] = df1['category_1'].str.replace('MusicalInstruments', 'Musical Instruments')
    df1['category_1'] = df1['category_1'].str.replace('HomeImprovement', 'Home Improvement')

    #Fixing Strings in Category_2 column
    df1['category_2'] = df1['category_2'].str.replace('&', ' & ')
    df1['category_2'] = df1['category_2'].str.replace(',', ', ')
    df1['category_2'] = df1['category_2'].str.replace('HomeAppliances', 'Home Appliances')
    df1['category_2'] = df1['category_2'].str.replace('AirQuality', 'Air Quality')
    df1['category_2'] = df1['category_2'].str.replace('WearableTechnology', 'Wearable Technology')
    df1['category_2'] = df1['category_2'].str.replace('NetworkingDevices', 'Networking Devices')
    df1['category_2'] = df1['category_2'].str.replace('OfficePaperProducts', 'Office Paper Products')
    df1['category_2'] = df1['category_2'].str.replace('ExternalDevices', 'External Devices')
    df1['category_2'] = df1['category_2'].str.replace('DataStorage', 'Data Storage')
    df1['category_2'] = df1['category_2'].str.replace('HomeStorage', 'Home Storage')
    df1['category_2'] = df1['category_2'].str.replace('HomeAudio', 'Home Audio')
    df1['category_2'] = df1['category_2'].str.replace('GeneralPurposeBatteries', 'General Purpose Batteries')
    df1['category_2'] = df1['category_2'].str.replace('BatteryChargers', 'Battery Chargers')
    df1['category_2'] = df1['category_2'].str.replace('CraftMaterials', 'Craft Materials')
    df1['category_2'] = df1['category_2'].str.replace('OfficeElectronics', 'Office Electronics')
    df1['category_2'] = df1['category_2'].str.replace('PowerAccessories', 'Power Accessories')
    df1['category_2'] = df1['category_2'].str.replace('CarAccessories', 'Car Accessories')
    df1['category_2'] = df1['category_2'].str.replace('HomeMedicalSupplies', 'Home Medical Supplies')
    df1['category_2'] = df1['category_2'].str.replace('HomeTheater', 'Home Theater')

    # Removing Whitespace from product_id
    df1['product_id'].str.strip()
    #Creating Categories for Rankings
    rating_score = []

    for score in df1['rating']:
        if score < 2.0 : rating_score.append('Poor')
        elif score < 3.0 : rating_score.append('Below Average')
        elif score < 4.0 : rating_score.append('Average')
        elif score < 5.0 : rating_score.append('Above Average')
        elif score == 5.0 : rating_score.append('Excellent')
    
    #Creating A new Column and Changing the Data Type
    df1['rating_score'] = rating_score
    df1['rating_score'] = df1['rating_score'].astype('category')

    #Reordered Categories
    df1['rating_score'] = df1['rating_score'].cat.reorder_categories(['Below Average', 'Average', 'Above Average', 'Excellent'], ordered=True)
    return df1


# Sentiment Breakdown by Selected Category Level
def sentiment_breakdown_by_category(df, category_level='category_1'):
    """
    Perform sentiment analysis using VADER and return sentiment breakdown for each category level.
    
    Parameters:
        df (DataFrame): The cleaned dataset.
        category_level (str): The category level to group by ('category_1' or 'category_2').
    
    Returns:
        dict: Dictionary of average sentiment percentages for each category.
    """
    # Initialize VADER Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Add sentiment scores for each review
    df[['pos', 'neu', 'neg', 'compound']] = df['review_content'].apply(
        lambda x: pd.Series(sia.polarity_scores(x if pd.notna(x) else ""))
    )

    # Group sentiment scores by the selected category level
    sentiment_by_category = df.groupby(category_level)[['pos', 'neu', 'neg']].mean() * 100

    # Convert to dictionary format for each category
    sentiment_breakdown = sentiment_by_category.to_dict(orient='index')

    return sentiment_breakdown

def preprocess_reviews(df):
    """
    Preprocess the reviews by cleaning, tokenizing, and lemmatizing the text.
    
    Parameters:
        df (DataFrame): DataFrame containing the review_content column.
    
    Returns:
        DataFrame: DataFrame with cleaned reviews.
    """
    st.write("Preprocessing reviews...")
    
    def clean_text(text):
        doc = nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2
        ]
        return ' '.join(tokens)
    
    # Apply cleaning to each review_content
    df['cleaned_review'] = df['review_content'].apply(lambda x: clean_text(x) if pd.notna(x) else "")
    return df

def perform_topic_modeling(cleaned_texts, num_topics=5):
    """
    Perform LDA topic modeling and return the visualization.
    
    Parameters:
        cleaned_texts (Series): Series containing cleaned review texts.
        num_topics (int): Number of topics to extract.
    
    Returns:
        pyLDAvis HTML object.
    """
    st.write("Performing topic modeling...")
    
    # Convert cleaned reviews to Bag of Words
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    review_matrix = vectorizer.fit_transform(cleaned_texts)
    
    # Perform Latent Dirichlet Allocation (LDA)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(review_matrix)
    
    # Prepare LDA visualization
    lda_vis = pyLDAvis.lda_model.prepare(lda, review_matrix, vectorizer)
    return pyLDAvis.prepared_data_to_html(lda_vis)

def extract_negative_sentences(text, sia):
    """
    Extract and return negative sentences from a given text.

    Args:
        text (str): The input text to analyze.
        sia (SentimentIntensityAnalyzer): An instance of VADER's SentimentIntensityAnalyzer used for sentiment analysis.

    Returns:
        str: A string containing all negative sentences concatenated together.
    """    
    sentences = nltk.sent_tokenize(text)
    negative_sentences = [s for s in sentences if sia.polarity_scores(s)['compound'] < -0.2]
    return " ".join(negative_sentences)

def process_customer_pain_points(df, category_level='category_1'):
    """
    Perform topic modeling on negative review content grouped by categories.
    
    Parameters:
        df (DataFrame): Cleaned dataframe.
        category_level (str): The category level to group by ('category_1' or 'category_2').
    
    Returns:
        tuple: (LDA model, transformed data, CountVectorizer, category groups).
    """
    sia = SentimentIntensityAnalyzer()

    # Extract negative sentences
    df['negative_content'] = df['review_content'].apply(lambda x: extract_negative_sentences(str(x), sia))

    # Group negative content by the selected category
    category_groups = df.groupby(category_level)['negative_content'].apply(lambda x: " ".join(x)).reset_index()

    # Vectorize the text
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords, ngram_range=(2, 3))
    transformed_data = vectorizer.fit_transform(category_groups['negative_content'])

    # Perform LDA topic modeling
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(transformed_data)

    return lda_model, transformed_data, vectorizer, category_groups


# Home page
def display_home():
    """
    Display the home page of the Amazon Product Review Sentiment Analysis web app.
    
    No inputs or return values; directly renders content in Streamlit.
    """
    st.title("🛍️ Amazon Product Review Sentiment Analysis")
    st.markdown(
        """
        Welcome to the Machine Learning Web App! This application leverages powerful natural language processing techniques to analyze Amazon customer review data and provide actionable insights. 

        ### The goal is to help businesses:
        - 🟢 Sentiment breakdown of reviews (positive, neutral, negative)
        - 🔑 Extract key themes from customer reviews
        - ⚠️ Identify customer pain points

        Explore the features to gain valuable insights for your business success.
        """
    )

def display_sentiment_breakdown(df, category_level):
    """
    Display the sentiment breakdown for product reviews based on selected category levels.

    Args:
        df (pd.DataFrame): The input DataFrame containing review data with 'category_1' and 'category_2'.
        category_level (str): The category level for grouping ('category_1' or 'category_2').

    Returns:
        None: The function renders the sentiment breakdown directly in the Streamlit app.
    """
     # **Add selection box to choose between category_1 and category_2**
     # Display unique values for category_1 and category_2
    category_1_values = df['category_1'].unique().tolist()
    category_2_values = df['category_2'].unique().tolist()

    # Display the unique category lists
    st.markdown("### Available Categories")
    with st.expander("View Available Categories"):
        st.write(f"**category_1 (Main Categories):**")
        st.markdown(", ".join(category_1_values))

        st.write(f"**category_2 (Sub-Categories):**")
        st.markdown(", ".join(category_2_values))

    category_level = st.selectbox("Choose Category Level:", ["category_1", "category_2"])
    sentiment_data = sentiment_breakdown_by_category(df, category_level=category_level)

    st.header(f"📊 Sentiment Breakdown by {category_level}")
    for category, sentiments in sentiment_data.items():
        st.subheader(f"Category: {category}")
        st.write(f"Positive Sentiment: {sentiments['pos']:.2f}%")
        st.write(f"Neutral Sentiment: {sentiments['neu']:.2f}%")
        st.write(f"Negative Sentiment: {sentiments['neg']:.2f}%")
        st.write("---")

    
# Function to display key phrases and themes
def display_key_phrases_and_themes(df):
    """
    Display key phrases and themes from Amazon product reviews using topic modeling.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'review_content' column.
    
    Returns:
        None: The function renders the topic modeling visualization directly in Streamlit.
    """
    st.header("🔑 Key Phrases and Themes (Topic Modeling)")
    
    # Preprocess reviews
    cleaned_df = preprocess_reviews(df)
    
    # Perform topic modeling and get HTML visualization
    lda_html = perform_topic_modeling(cleaned_df['cleaned_review'])
    
    # Display interactive topic modeling visualization
    st.components.v1.html(lda_html, width=1300, height=800)
    

def display_customer_pain_points(df):
    """
    Display customer pain points using topic modeling to analyze negative reviews by category.

    Args:
        df (pd.DataFrame): DataFrame containing review data with 'category_1', 'category_2', and 'review_content'.

    Returns:
        None: The function renders the customer pain points visualization directly in Streamlit.
    """
    # **Add selection box to choose between category_1 and category_2**
    # Display unique values for category_1 and category_2
    category_1_values = df['category_1'].unique().tolist()
    category_2_values = df['category_2'].unique().tolist()

    # Display the unique category lists
    st.markdown("### Available Categories")
    with st.expander("View Available Categories"):
        st.write(f"**category_1 (Main Categories):**")
        st.markdown(", ".join(category_1_values))

        st.write(f"**category_2 (Sub-Categories):**")
        st.markdown(", ".join(category_2_values))
    category_level = st.selectbox("Choose Category Level:", ['category_1', 'category_2'])
    st.header(f"⚠️ Customer Pain Points by {category_level}")

    lda_model, transformed_data, vectorizer, category_groups = process_customer_pain_points(df, category_level)

    # Ensure `vectorizer` is the CountVectorizer object and not an array
    if not isinstance(vectorizer, CountVectorizer):
        raise ValueError("Expected `vectorizer` to be a CountVectorizer object, but got something else.")

    # Extract required inputs
    vocab = vectorizer.get_feature_names_out()  # Extract the vocabulary list
    term_frequency = np.array(transformed_data.sum(axis=0)).flatten()  # Frequency of each term
    doc_lengths = [len(text.split()) for text in category_groups['negative_content']]  # Words in each document


    pyldavis_data = pyLDAvis.lda_model.prepare(
        lda_model=lda_model,         # Fitted LDA model
        dtm=transformed_data,        # Pass the sparse matrix
        vectorizer=vectorizer        # Pass the original CountVectorizer object
    )

    # Display in Streamlit
    html_vis = pyLDAvis.prepared_data_to_html(pyldavis_data)
    components.html(html_vis, width=1200, height=800)

def main():
    # Load data (replace with your actual path)
    cleaned_df = preprocess_data(df)
    st.sidebar.title("Navigation")
    selected_option = st.sidebar.radio(
        "Choose a category:",
        ("Home", "Sentiment Breakdown", "Key Phrases and Themes", "Customer Pain Points")
    )

    # Navigation logic
    category_level = 'category_1'
    if selected_option == "Home":
        display_home()
    elif selected_option == "Sentiment Breakdown":
        display_sentiment_breakdown(cleaned_df, category_level)
    elif selected_option == "Key Phrases and Themes":
        display_key_phrases_and_themes(cleaned_df)
    elif selected_option == "Customer Pain Points":
        display_customer_pain_points(cleaned_df)

if __name__ == "__main__":
    main()

# cleaned_data = preprocess_data(df)
# print(cleaned_data['category_2'].unique())
