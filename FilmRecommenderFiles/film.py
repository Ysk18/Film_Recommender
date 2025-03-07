import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# STEP 1 - CREATE RATINGS MATRIX
with open("filtered.json") as f: # with open("filtered_reviews.json") as f:
    filtered_data = json.load(f)

# m x n ratings matrix
reviewers = sorted(set(entry['reviewer'] for entry in filtered_data))
movies = sorted(set(entry['movie'] for entry in filtered_data))

reviewer_to_index = {reviewer : i for i, reviewer in enumerate(reviewers)}
movie_to_index = {movie : i for i, movie in enumerate(movies)}

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

# STEP 5 - PREDICTIONS & MAE FUNCTIONS (already correct)

def calculating_prediction(U, S, Vt, averages_reviewer, user_a, item_j, reduce_k, neighbor_L):
    U_red = reduction_U(U, reduce_k)
    VT_red = reduction_VT(Vt, reduce_k)
    S_red = reduction_S(S, reduce_k)

    R_reduced = np.dot(np.dot(U_red, S_red), VT_red)

    meta_ratings = np.dot(np.sqrt(S_red), VT_red)

    # calculating similarity values for the user a
    sim_values = []
    for f in range(VT_red.shape[1]):
        sim_above = np.dot(meta_ratings[:, item_j], meta_ratings[:, f])
        sim_below = np.linalg.norm(meta_ratings[:, item_j]) * np.linalg.norm(meta_ratings[:, f])
        sim_j = sim_above / sim_below if sim_below != 0 else 0
        sim_values.append((f, sim_j))

    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)[:neighbor_L] # reversed

    pre_above = sum(sim[1] * (R_reduced[user_a, sim[0]] + averages_reviewer[user_a]) for sim in sim_values)
    pre_below = sum(np.abs(sim[1]) for sim in sim_values)
    prediction = pre_above / pre_below if pre_below != 0 else 0
    #if prediction > 5 : print(prediction)
    return prediction

# MAE Calculation
def calculate_mae_per_user(R, row_averages, U, S, Vt, neighbor_L_values, reduce_k_values, user):
    total_MAE_values = {}
    user_items = np.nonzero(R[user])[0].tolist()

    output_cache = {}
    for neighbor_L in neighbor_L_values:
        for reduce_k in reduce_k_values:
            key = (reduce_k, neighbor_L)
            predictions = [calculating_prediction(U, S, Vt, row_averages, user, movie, reduce_k, neighbor_L) for movie in user_items]
            output_cache[key] = predictions

    for key, predictions in output_cache.items():
        total_error = sum(abs(predictions[i] - R[user, user_items[i]]) for i in range(len(user_items)))
        total_MAE_values[key] = total_error / len(user_items) if len(user_items) > 0 else 0

    return total_MAE_values

def calculate_mae_all_users(R, row_averages, U, S, Vt, neighbor_L_values, reduce_k_values):
    total_MAE_values = {}
    user_indices = np.where(np.any(R != 0, axis=1))[0]

    for user in user_indices:
        user_MAE_values = calculate_mae_per_user(R, row_averages, U, S, Vt, neighbor_L_values, reduce_k_values, user)
        for key, mae in user_MAE_values.items():
            if key not in total_MAE_values:
                total_MAE_values[key] = 0
            total_MAE_values[key] += mae

    average_MAE_values = {key: mae / len(user_indices) for key, mae in total_MAE_values.items()}
    return average_MAE_values

def plot_results(mae_values, reduce_k_values, neighbor_L_values, title):
    for neighbor_L in neighbor_L_values:
        result_values = [mae_values[(reduce_k, neighbor_L)] for reduce_k in reduce_k_values]
        plt.plot(reduce_k_values, result_values, label=f'neighbor_L={neighbor_L}')
    
    plt.xticks(reduce_k_values)
    plt.xlabel('reduce_k')
    plt.ylabel('MAE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# MAIN EXECUTION

# Normalize the matrix
R_norm, row_averages, _ = normalization(R, num_reviewers, num_movies)

# Perform SVD
U, S, Vt = apply_svd(R_norm)

# Neighbor and K values
neighbor_L_values = [20, 60, 80]
reduce_k_values = [2, 4, 6, 8, 10, 20]

# Calculate MAE for all users
average_MAE_values = calculate_mae_all_users(R, row_averages, U, S, Vt, neighbor_L_values, reduce_k_values)
plot_results(average_MAE_values, reduce_k_values, neighbor_L_values, "MAE for All Users")

# Calculate MAE for a specific user (e.g., user_a1 = 1)
user_a1 = 1
user_1_MAE_values = calculate_mae_per_user(R, row_averages, U, S, Vt, neighbor_L_values, reduce_k_values, user_a1)
plot_results(user_1_MAE_values, reduce_k_values, neighbor_L_values, f"MAE for User {user_a1}")



"""
SVD Calculation without using svd functions
QR Algorithm used

def qr_algorithm(A, num_iterations=500): # QR algorithm with certain iterations 
                                        # to find eigenvalues and vectors
    n = A.shape[0]
    
    Q = np.eye(n) # Initializing Q and R matrices
    R = A.copy()
    m,n = Q.shape
    new_matrix = np.eye(m, n)

    for i in range(num_iterations):
        
        Q, R = np.linalg.qr(R @ Q) # QR decomposition
        new_matrix = np.dot(new_matrix, Q)    

    eigenvalues = np.diagonal(R) 

    eigenvectors = new_matrix

    # print(eigenvalues) # for checking

    return eigenvalues, eigenvectors


def svd_without_eig(A):
    
    ATA = np.dot(A.T, A) # A.T * A
    
    eigenvalues, V = qr_algorithm(ATA) # eigenvalues and V
    
    singular_values = np.sqrt(np.abs(eigenvalues)) # singular values
    S = np.zeros_like(A)
    np.fill_diagonal(S, singular_values) # singular value matrix
    
    U = np.dot(A, V) / singular_values # U
    U = U[:, :min(A.shape)]
    
    return U, S, V


U, S, V = svd_without_eig(matrix_norm)
"""