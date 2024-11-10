# Movie Recommendation System

## Project Overview
In this project, a Movie Recommendation System is built using the **TMDB 5000 Movie Dataset**. Recommendation systems are essential tools in today’s data-driven world, widely used by companies like Netflix, Amazon, and YouTube. This project focuses on building a baseline movie recommender that predicts movie ratings and preferences, enabling users to receive personalized recommendations based on various filtering techniques. The system uses **Collaborative Filtering**, **Content-Based Filtering**, and **Demographic Filtering** to make movie suggestions.

## Project Objectives
- Develop a recommendation system using different filtering techniques: Demographic, Content-Based, and Collaborative Filtering.
- Implement and evaluate models using item-item similarity matrix, matrix factorization, and feature engineering techniques.
- Predict user ratings for movies and provide personalized movie suggestions.
  
## Key Steps

### Data Collection
- Integrated over **100k+ ratings** data of **5000+ movies** from multiple CSV files into a single DataFrame.

### Data Preprocessing
- Cleaned and preprocessed the movie dataset for use in recommendation algorithms.
- Handled missing data and outliers for better model performance.

### Recommendation Techniques
- **Demographic Filtering**: Recommended movies based on the popularity and genre, targeting users with similar demographic features.
- **Content-Based Filtering**: Suggested similar movies based on metadata like genre, director, and actors.
- **Collaborative Filtering**: Matched users with similar interests and provided recommendations based on user interaction patterns. 

### Model Training and Evaluation
- Built models using **Item-Item Similarity Matrix**, **Matrix Factorization** (SVD), and **Content-Based Filtering** using feature engineering.
- Evaluated model performance using **RMSE (Root Mean Squared Error)**, achieving a score of **0.89** for predicted ratings.

## Tools and Technologies Used
- **Languages**: Python
- **Development Tools**: Jupyter Notebook, Spyder IDE
- **Libraries**:
  - **Data Processing**: Pandas, NumPy
  - **Recommendation Systems**: Scikit-Learn, Surprise, TensorFlow
  - **Data Visualization**: Matplotlib, Seaborn

## Results
- **Predicted Movie Ratings**: Achieved a **RMSE of 0.89** for predicted ratings using **SVD** and the latent factor model.
- **Recommendation Accuracy**: Successfully built recommendation models using collaborative and content-based methods.
- **Scalable System**: The recommendation system can be extended to work with larger datasets and integrate with real-time systems.

## Sample Code Snippet

```python
# Importing necessary libraries
from sklearn.decomposition import TruncatedSVD
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# Example for Collaborative Filtering with Matrix Factorization using SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# SVD model
svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)

# Example for Content-Based Filtering (using genres as features)
movie_genres = movie_metadata[['movieId', 'genres']]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movie_genres['genres'])

# Item-Item similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/Ravii-saini/Movie_Recommendation_System.git>
   ```

2. **Install Required Packages**:
   Install the necessary libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn surprise tensorflow matplotlib seaborn
   ```

3. **Run the Notebook**:
   Open the project in Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Future Improvements
- Implement advanced models like **Neural Collaborative Filtering (NCF)** for better accuracy.
- Integrate real-time movie recommendation features using **streaming data**.
- Use **hybrid models** by combining content-based and collaborative filtering methods for more personalized recommendations.
- Expand the dataset to include user reviews, ratings, and additional movie metadata for better predictions.

## License
This project is intended for educational purposes and self-learning.
```

