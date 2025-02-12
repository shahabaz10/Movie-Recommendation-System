# Movie-Recommendation-System
 Built a user-based and item-based collaborative and content filtering model using Kaggle’s Movies and Ratings dataset
This project builds a recommendation system using two different approaches:

Collaborative Filtering: Uses user-item interactions to suggest movies.

Content-Based Filtering: Recommends movies based on metadata (genre, actors, etc.).

The dataset is sourced from Kaggle's Movies & Ratings dataset.

Features

Collaborative Filtering: Predicts user preferences using similarity-based filtering.

Content-Based Filtering: Suggests movies based on metadata similarity.

Hybrid Model: Combines both techniques to improve recommendations.

Technologies Used

Python (Pandas, NumPy, Matplotlib)

Scikit-Learn (Cosine Similarity, TF-IDF Vectorization)

Surprise Library (SVD, ALS)

Jupyter Notebook

How to Run

Clone the repository.

Install dependencies using pip install -r requirements.txt.

Run MRS_collaborative.ipynb for collaborative filtering.

Run MRS_content_based.ipynb for content-based filtering.
