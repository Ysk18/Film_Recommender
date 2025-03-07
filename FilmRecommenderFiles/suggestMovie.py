import json
import numpy as np
# import pandas as pd
import os

"""
# Define the directory where your JSON files are located
data_directory = "./"  # Adjust this path if needed

# Initialize an empty list to hold all data
all_data = []

# Load data from each JSON file
for i in range(1, 2):  # Assuming you have part-01.json to part-06.json
    file_path = os.path.join(data_directory, f"part-0{i}.json")
    with open(file_path) as f:
        data = json.load(f)
        all_data.extend(data)  # Add all data to the combined list

# Count reviews per reviewer
reviewer_counts = {}

for review in all_data:
    reviewer = review["reviewer"]
    reviewer_counts[reviewer] = reviewer_counts.get(reviewer, 0) + 1

# Filter data based on reviewer count
filtered_data = []

for entry in all_data:
    reviewer = entry["reviewer"]
    review_count = reviewer_counts[reviewer]
    
    # Check if the reviewer has made at least 20 reviews
    if review_count >= 20:
        movie = entry["movie"]
        rating = entry["rating"]
        # Add the review to the filtered dataset
        filtered_data.append({'reviewer': reviewer, 'movie': movie, 'rating': rating})

# Save the filtered dataset to a new JSON file
with open('filtered_reviews.json', 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)

"""



# STEP 1 - CREATE RATINGS MATRIX
with open("filtered.json") as f: #filtered_reviews.json so big that it takes lots of time
    filtered_data = json.load(f)

# m x n ratings matrix
reviewers = sorted(set(entry['reviewer'] for entry in filtered_data))
movies = sorted(set(entry['movie'] for entry in filtered_data))

reviewer_to_index = {reviewer: i for i, reviewer in enumerate(reviewers)}
movie_to_index = {movie: i for i, movie in enumerate(movies)}
index_to_movie = {v: k for k, v in movie_to_index.items()}  # Reverse mapping for suggestions

num_reviewers = len(reviewers)  # m users
num_movies = len(movies)  # n movies

matrix = np.zeros((num_reviewers, num_movies))

# fill the matrix
for entry in filtered_data:
    reviewer_index = reviewer_to_index[entry['reviewer']]
    movie_index = movie_to_index[entry['movie']]
    rating = entry['rating']
    if rating is not None:
        matrix[reviewer_index, movie_index] = rating

ratings_matrix = matrix.copy()
R = ratings_matrix.copy()

# STEP 2 - NORMALIZE RATINGS MATRIX
def normalization(ratings_matrix, num_reviewers, num_movies):
    col_averages = []
    row_averages = []

    # Calculate column averages (per movie)
    for j in range(num_movies):
        temp_sum = np.sum(ratings_matrix[:, j])
        col_averages.append(temp_sum / num_reviewers)

    # Calculate row averages (per reviewer)
    for i in range(num_reviewers):
        temp_sum = np.sum(ratings_matrix[i, :])
        row_averages.append(temp_sum / num_movies)

    matrix_norm = np.copy(ratings_matrix)
    for i in range(num_reviewers):
        for j in range(num_movies):
            if matrix_norm[i][j] == 0:
                matrix_norm[i][j] += col_averages[j]
            matrix_norm[i][j] -= row_averages[i]

    return matrix_norm, row_averages, col_averages

# STEP 3 - SVD
def apply_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(S)
    return U, S, Vt

# STEP 4 - DIMENSION REDUCTION FUNCTIONS (already correct)
def reduction_U(matrix, k):
    new_matrix = matrix[:, :k]
    return new_matrix

def reduction_VT(matrix, k):
    new_matrix = matrix[:k, :]
    return new_matrix

def reduction_S(matrix, k):
    new_matrix = matrix[:k, :k]
    return new_matrix

# STEP 5 - PREDICTION AND MOVIE SUGGESTION FUNCTIONS
def suggest_movies_multiple(U, S, Vt, movie_to_index, index_to_movie, user_movies, user_ratings, top_n=10):
    """
    Suggests movies based on multiple user-rated movies.
    """
    recommended_movies = {}
    
    for user_movie, user_rating in zip(user_movies, user_ratings):
        if user_movie not in movie_to_index:
            print(f"Movie '{user_movie}' not found in the dataset.")
            continue
        
        # Get index of the movie rated by the user
        movie_index = movie_to_index[user_movie]
        
        # Perform the SVD reduction to get the reduced ratings matrix
        U_red = reduction_U(U, 20)  # You can adjust 'k' here, let's use 20 for reduction
        VT_red = reduction_VT(Vt, 20)
        S_red = reduction_S(S, 20)
        
        # Reconstruct the ratings matrix
        R_reduced = np.dot(np.dot(U_red, S_red), VT_red)
        
        # Get the feature vector for the user-rated movie
        user_movie_features = R_reduced[:, movie_index]
        
        # Calculate similarity scores between user_movie and all other movies
        for i in range(R_reduced.shape[1]):
            if i != movie_index:
                sim_numerator = np.dot(user_movie_features, R_reduced[:, i])
                sim_denominator = np.linalg.norm(user_movie_features) * np.linalg.norm(R_reduced[:, i])
                sim = sim_numerator / sim_denominator if sim_denominator != 0 else 0
                
                if i not in recommended_movies:
                    recommended_movies[i] = sim
                else:
                    recommended_movies[i] += sim
    
    # Sort the movies by aggregated similarity score in descending order
    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations (excluding movies the user has already rated)
    recommendations = []
    rated_movie_indices = [movie_to_index[movie] for movie in user_movies]
    
    for i, (movie_idx, sim_score) in enumerate(recommended_movies):
        if movie_idx not in rated_movie_indices:
            recommendations.append(index_to_movie[movie_idx])
            if len(recommendations) >= top_n:
                break
    
    return recommendations

def print_top_20_movies(ratings_matrix, movie_to_index, index_to_movie, min_ratings):
    num_movies = ratings_matrix.shape[1]
    movie_ratings = []

    # Calculate the average rating and the number of ratings for each movie
    for j in range(num_movies):
        movie_ratings_sum = np.sum(ratings_matrix[:, j])
        num_ratings = np.count_nonzero(ratings_matrix[:, j])
        
        if num_ratings >= min_ratings:
            average_rating = movie_ratings_sum / num_ratings
            movie_ratings.append((index_to_movie[j], average_rating, num_ratings))

    # Sort movies by their average rating (highest first)
    movie_ratings_sorted = sorted(movie_ratings, key=lambda x: x[1], reverse=True)

    print("\nTop 20 Movies by Average Rating (with at least 10 ratings): \n   Warning: If dataset is small you can see less than 20 movies!")
    for i, (movie, avg_rating, num_ratings) in enumerate(movie_ratings_sorted[:20]):
        print(f"{i+1}. {movie} - Average Rating: {avg_rating:.2f} (Rated by {num_ratings} users)")

print_top_20_movies(ratings_matrix, movie_to_index, index_to_movie, min_ratings=10)




def suggest_movie_names(partial_name, movie_to_index):
    """
    Suggests movie names based on the partial input provided by the user.
    """
    suggestions = []
    for movie in movie_to_index.keys():
        if partial_name.lower() in movie.lower():  # Case insensitive match
            suggestions.append(movie)
    return suggestions

def get_number_of_movies():
    while True:
        try:
            n = int(input("How many movies would you like to rate? "))
            if n > 0:
                return n
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Function to get user input and recommend movies
def recommend_movies_based_on_multiple_inputs(U, S, Vt, movie_to_index, index_to_movie):
    # Print top 20 rated movies before asking for user input

    n = get_number_of_movies()
    
    user_movies = []
    user_ratings = []
    
    for i in range(n):
        while True:
            movie_input = input(f"Enter movie {i+1} you've watched: ").strip()  # Take input
            suggestions = suggest_movie_names(movie_input, movie_to_index)  # Get suggestions
            
            if suggestions:
                print("Did you mean:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")
            
            if movie_input.lower() in {k.lower() for k in movie_to_index.keys()}:
                rating = float(input(f"Rate {movie_input} out of 10: "))
                # Append the original movie name (not the normalized one)
                original_movie_name = [k for k, v in movie_to_index.items() if k.strip().lower() == movie_input.lower()][0]
                user_movies.append(original_movie_name)
                user_ratings.append(rating)
                break  # Exit the loop when a valid movie is found and rated
            else:
                print(f"Movie '{movie_input}' not found in the dataset. Please try another movie.")
    
    # Suggest 10 movies based on all inputs
    recommendations = suggest_movies_multiple(U, S, Vt, movie_to_index, index_to_movie, user_movies, user_ratings, top_n=10)
    
    if recommendations:
        print("\nBased on your ratings, we recommend these 10 movies:")
        for i, movie in enumerate(recommendations):
            print(f"{i+1}. {movie}")
    else:
        print("Sorry, no recommendations could be made.")


# MAIN EXECUTION
# Normalize the matrix
R_norm, row_averages, _ = normalization(R, num_reviewers, num_movies)

# Perform SVD
U, S, Vt = apply_svd(R_norm)

# Recommend movies based on user input
recommend_movies_based_on_multiple_inputs(U, S, Vt, movie_to_index, index_to_movie)