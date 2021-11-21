
import pandas as pd
import numpy as np
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# reading the data from the preprocessed .csv file
data = pd.read_csv('preprocessed_data.csv')

# making the new column containing combination of all the features
  data["combination"] = data["plot"]+" "+data["genre"]+" "+data["Directed by"]+" "+data["Produced by"]+" "+data["Starring"]+" "+data["Music by"]+" "+data["Distributed by"]


# creating a tfidf matrix
tvf = TfidfVectorizer(min_df = 3, max_features = None, 
                         strip_accents = "unicode", analyzer = "word",
                         token_pattern=r'\w{1,}',ngram_range=(1, 3),
                         stop_words = 'english')
    
tvf_matrix = tvf.fit_transform(data["combination"])


# creating a similarity score matrix
sim = cosine_similarity(tvf_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('preprocessed_data.csv',index=False)
