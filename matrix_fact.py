import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class MatrixFactorization :
    def __init__(self, n_users, n_items, n_factors=20, regularization=1):
        """
        Initialize the matrix factorization model

        Parameters :
        -----------
        n_users : int
        Number of users in the dataset
        n_items : int
        Number of items in the dataset
        n_factors : int
        Number of latent factors ( embedding dimension )
        regularization : float
        Regularization parameter to avoid overfitting
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.regularization = regularization

        # Initialize user and item matrices with small random values
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, n_factors))

        # Initialize bias terms. By setting these to zero, the formula reduces to: r = p times q transpose
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def predict(self, user_id, item_id):
        """
        Predict the rating for a user - item pair

        Parameters :
        -----------
        user_id : int
        User ID
        item_id : int
        Item ID

        Returns :
        --------
        float
        Predicted rating
        """
        # Check if user_id or item_id is valid
        if user_id >= self.n_users or item_id >= self.n_items :
            return self.global_bias

        # Compute the predicted rating
        prediction = (
            self.global_bias
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + np.dot(self.user_factors[user_id], self.item_factors[item_id])
        )

        return np.clip(prediction, 1.0, 5.0)  # Clip to rating scale [1, 5] 

    def fit_als(self, ratings, n_iterations=10, test_data=None, early_stopping=True, patience=2):
        """
        Train the model using Alternating Least Squares

        Parameters :
        -----------
        ratings : pandas DataFrame
            Dataframe containing (user_id, item_id, rating) tuples
        n_iterations : int
            Number of iterations to run
        test_data : pandas DataFrame, optional
            Test data to evaluate model during training
        early_stopping : bool
            Whether to use early stopping to prevent overfitting
        patience : int
            Number of iterations to wait before early stopping if test error increases

        Returns :
        --------
        tuple
            (training_errors, test_errors) - Lists of training and test errors at each iteration
        """

        # TODO : TASK 1 - Implement the ALS algorithm
        # Extract data
        user_ids = ratings['user_id'].values
        item_ids = ratings['item_id'].values
        rating_values = ratings['rating'].values

        # Calculate global bias (mean of all ratings)
        self.global_bias = np.mean(rating_values)

        # Create user-item rating matrix (sparse representation)
        user_item_matrix = {}
        for u, i, r in zip(user_ids, item_ids, rating_values):
            if u not in user_item_matrix:
                user_item_matrix[u] = {}
            user_item_matrix[u][i] = r

        # Store errors for each iteration
        training_errors = []
        test_errors = []
        
        # Variables for early stopping
        best_test_rmse = float('inf')
        counter = 0
        best_model_state = None

        for iteration in range(n_iterations):
            # Fix item factors, solve for user factors
            for u in range(self.n_users):
                # 1. Get items rated by user u
                # 2. If user has no ratings, skip
                if u not in user_item_matrix:
                    continue
                rated_items = user_item_matrix[u]
                if len(rated_items) == 0:
                    continue

                # 3. Build the system A * x = b for user factors
                A = np.zeros((self.n_factors, self.n_factors))
                b = np.zeros(self.n_factors)

                for i, r in rated_items.items():
                    # Update A
                    A += np.outer(self.item_factors[i], self.item_factors[i])

                    # Update b
                    residual = r - self.global_bias - self.user_bias[u] - self.item_bias[i]
                    b += residual * self.item_factors[i]

                # Add regularization
                A += self.regularization * np.eye(self.n_factors)
                
                # 4. Solve the system using np.linalg.solve
                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Handle singular matrix issues
                    self.user_factors[u] = np.zeros(self.n_factors)
                
                # Update user bias
                numerator = sum(
                    r - self.global_bias - self.item_bias[i] - np.dot(self.user_factors[u], self.item_factors[i])
                    for i, r in rated_items.items()
                )
                self.user_bias[u] = numerator / (len(rated_items) + self.regularization)

                pass

            # Fix user factors, solve for item factors
            for i in range(self.n_items):
                # 1. Get users who rated item i
                rated_by_users = {}
                for u, items in user_item_matrix.items():
                    if i in items:
                        rated_by_users[u] = items[i]
               
                if len(rated_by_users) == 0:
                    continue

                # 2. Build the system A * x = b for item factors
                A = np.zeros((self.n_factors, self.n_factors))
                b = np.zeros(self.n_factors)

                for u, r in rated_by_users.items():
                    # Update A
                    A += np.outer(self.user_factors[u], self.user_factors[u])

                    # Update b
                    residual = r - self.global_bias - self.user_bias[u] - self.item_bias[i]
                    b += residual * self.user_factors[u]

                # Add regularization
                A += self.regularization * np.eye(self.n_factors)
                
                # 3. Solve the system
                try:
                    self.item_factors[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Handle singular matrix issues
                    self.item_factors[i] = np.zeros(self.n_factors)

                # Update item bias
                numerator = sum(
                    r - self.global_bias - self.user_bias[u] - np.dot(self.user_factors[u], self.item_factors[i])
                    for u, r in rated_by_users.items()
                )
                self.item_bias[i] = numerator / (len(rated_by_users) + self.regularization)
                
                pass

            # Calculate training error for this iteration
            train_rmse = self.calculate_rmse(ratings)
            training_errors.append(train_rmse)
            
            # Calculate test error if test data is provided
            test_rmse = None
            if test_data is not None:
                test_rmse = self.calculate_rmse(test_data)
                test_errors.append(test_rmse)
                print(f"Iteration {iteration + 1}/{n_iterations}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                
                # Early stopping check
                if early_stopping:
                    if test_rmse < best_test_rmse:
                        best_test_rmse = test_rmse
                        counter = 0
                        # Save the best model state
                        best_model_state = {
                            'user_factors': self.user_factors.copy(),
                            'item_factors': self.item_factors.copy(),
                            'user_bias': self.user_bias.copy(),
                            'item_bias': self.item_bias.copy(),
                            'global_bias': self.global_bias
                        }
                    else:
                        counter += 1
                    
                    # If test error hasn't improved for 'patience' iterations, stop training
                    if counter >= patience:
                        print(f"Early stopping at iteration {iteration + 1}. Best test RMSE: {best_test_rmse:.4f}")
                        # Restore best model state
                        if best_model_state:
                            self.user_factors = best_model_state['user_factors']
                            self.item_factors = best_model_state['item_factors']
                            self.user_bias = best_model_state['user_bias']
                            self.item_bias = best_model_state['item_bias']
                            self.global_bias = best_model_state['global_bias']
                        break
            else:
                print(f"Iteration {iteration + 1}/{n_iterations}, Train RMSE: {train_rmse:.4f}")

        return training_errors, test_errors
    
    def calculate_rmse(self, true_ratings):
        """
        Calculate RMSE on a set of true ratings

        Parameters :
        -----------
        true_ratings : pandas DataFrame
        Dataframe containing ( user_id , item_id , rating ) tuples

        Returns :
        --------
        float
        Root Mean Square Error
        """ 
        user_ids = true_ratings['user_id'].values
        item_ids = true_ratings['item_id'].values
        true_rating_values = true_ratings['rating'].values
    
        # Get predictions for all ratings
        predicted_ratings = np.array([self.predict(u, i) for u, i in zip(user_ids, item_ids)])

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_rating_values, predicted_ratings))

        return rmse

    def get_user_recommendations(self, user_id, n_recommendations=10, 
                                 exclude_rated=True, rated_items=None):
        """
        Get top N recommendations for a specific user

        Parameters :
        -----------
        user_id : int
        User ID
        n_recommendations : int
        Number of recommendations to return
        exclude_rated : bool
        Whether to exclude already rated items
        rated_items : list
        List of items already rated by the user

        Returns :
        --------
        list
        List of recommended item IDs with their predicted ratings
        """
        

        """
        Get top N recommendations for a specific user
        """
        # Check if user_id is valid
        if user_id >= self.n_users:
            return []
        
        # Get items rated by user if not provided
        if exclude_rated and rated_items is None:
            # This would require scanning through the data
            # For simplicity , we 'll just use the provided rated_items
            if rated_items is None:
                rated_items = []

        # Calculate predicted ratings for all items
        predictions = []
        for item_id in range(self.n_items):
            # Skip items the user has already rated
            if exclude_rated and item_id in rated_items:
                continue

            # Predict rating
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        # Sort by predicted rating in descending order
        predictions.sort(key = lambda x : x [1], reverse = True)

        # Return top N recommendations
        return predictions [: n_recommendations]
        
        pass


# Data loading function
def load_movielens_data(file_path='ml-100k/u.data'):
    """
    Load the MovieLens 100 K dataset

    Parameters :
    -----------
    file_path : str
        Path to the data file

    Returns :
    --------
    pandas DataFrame
    
    Dataframe containing user_id , item_id , rating , timestamp
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(file_path, sep='\t', names=column_names)

    # Convert to 0-based indexing
    data['user_id'] = data['user_id'] - 1
    data['item_id'] = data['item_id'] - 1

    return data

# Function to load movie titles
def load_movie_titles(file_path='ml-100k/u.item'):
    """
    Load movie titles from the MovieLens dataset

    Parameters :
    -----------
    file_path : str
    Path to the item file

    Returns :
    --------
    pandas DataFrame
    Dataframe containing item_id and title
    """
    movies = pd.read_csv(
        file_path,
        sep='|',
        encoding='latin-1',
        header=None,
        usecols=[0, 1],
        names=['item_id', 'title']
    )
    movies['item_id'] = movies['item_id'] - 1  # Convert to 0-based indexing
    return movies

def tune_regularization(train_data, validation_data, n_users, n_items, n_factors=20, 
                      reg_values=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                      n_iterations=10):
    """
    Find the optimal regularization parameter
    
    Parameters:
    -----------
    train_data : pandas DataFrame
        Training data
    validation_data : pandas DataFrame
        Validation data for tuning
    n_users : int
        Number of users
    n_items : int
        Number of items
    n_factors : int
        Number of latent factors
    reg_values : list
        List of regularization parameter values to try
    n_iterations : int
        Number of iterations for each model training
        
    Returns:
    --------
    tuple
        (best_reg, best_rmse, all_results) - Best regularization parameter,
        best RMSE, and dictionary with all results
    """
    results = {}
    best_rmse = float('inf')
    best_reg = None
    
    print("\nTuning regularization parameter...")
    
    for reg in reg_values:
        print(f"\nTrying regularization = {reg}")
        model = MatrixFactorization(n_users, n_items, n_factors=n_factors, regularization=reg)
        
        # Train the model
        _, _ = model.fit_als(train_data, n_iterations=n_iterations, 
                           test_data=validation_data, early_stopping=True, patience=2)
                           
        # Evaluate on validation set
        val_rmse = model.calculate_rmse(validation_data)
        results[reg] = val_rmse
        
        print(f"Regularization = {reg}, Validation RMSE = {val_rmse:.4f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_reg = reg
    
    print(f"\nBest regularization parameter: {best_reg} with validation RMSE: {best_rmse:.4f}")
    
    return best_reg, best_rmse, results

def plot_learning_curves(training_errors, validation_errors=None):
    """
    Plot learning curves to visualize overfitting
    
    Parameters:
    -----------
    training_errors : list
        List of training errors at each iteration
    validation_errors : list, optional
        List of validation errors at each iteration
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training curve
    plt.plot(range(1, len(training_errors) + 1), training_errors, 
             marker='o', linestyle='-', color='blue', label='Training RMSE')
    
    # Plot validation curve if available
    if validation_errors:
        plt.plot(range(1, len(validation_errors) + 1), validation_errors, 
                 marker='s', linestyle='-', color='red', label='Validation RMSE')
        
        # Add a horizontal line at the minimum validation error
        min_val_error = min(validation_errors)
        min_idx = validation_errors.index(min_val_error) + 1
        plt.axhline(y=min_val_error, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=min_idx, color='r', linestyle='--', alpha=0.3)
        
        # Annotate the minimum point
        plt.annotate(f'Best: {min_val_error:.4f}',
                    xy=(min_idx, min_val_error),
                    xytext=(min_idx + 1, min_val_error + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Check for overfitting
        if min_idx < len(validation_errors):
            # If the minimum is not at the last iteration, we might have overfitting
            overfitting_text = (f"Potential overfitting after iteration {min_idx}\n"
                                f"Training continues to improve while validation worsens")
            plt.text(0.5, 0.01, overfitting_text, 
                    horizontalalignment='center',
                    verticalalignment='bottom', 
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='yellow', alpha=0.2))
    
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Learning Curves: Training vs Validation Error')
    
    # Add gap between training and validation as annotation
    if validation_errors and len(training_errors) == len(validation_errors):
        last_train = training_errors[-1]
        last_val = validation_errors[-1]
        gap = last_val - last_train
        plt.annotate(f'Gap: {gap:.4f}',
                    xy=(len(training_errors), (last_train + last_val)/2),
                    xytext=(len(training_errors) + 0.5, (last_train + last_val)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    print(" Loading data ...")
    ratings = load_movielens_data()
    movies = load_movie_titles()

    # Some basic statistics
    n_users = ratings['user_id'].nunique()
    n_items = ratings['item_id'].nunique()
    print(f" Number of users: {n_users}, Number of items: {n_items}")
    print(f" Number of ratings: {len(ratings)}")
    
    # Create train/validation/test split
    # First split off 20% for final testing
    train_val_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    # Then split remaining data into train and validation
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 of original data
    
    print(f" Training set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")

    # Option 1: Hyperparameter tuning
    perform_tuning = True
    
    if perform_tuning:
        # Find the best regularization parameter
        best_reg, _, _ = tune_regularization(
            train_data, 
            val_data, 
            n_users, 
            n_items, 
            n_factors=20,
            reg_values=[0.01, 0.1, 1.0, 4.0, 10.0, 20.0],
            n_iterations=10
        )
        
        # Combine train and validation data for final model training
        final_train_data = pd.concat([train_data, val_data])
        
        # Create and train the final model with best parameters
        print("\nTraining final model with best regularization parameter...")
        final_model = MatrixFactorization(n_users, n_items, n_factors=20, regularization=best_reg)
        training_errors, test_errors = final_model.fit_als(
            final_train_data, 
            n_iterations=15,  # Train longer on final model
            test_data=test_data
        )
        
        # Evaluate on test set
        test_rmse = final_model.calculate_rmse(test_data)
        print(f"\nFinal Test RMSE: {test_rmse:.4f}")
        
        # Use this model for recommendations
        model = final_model
        
    else:
        # Option 2: Simple model without tuning
        print("\nInitializing model...")
        model = MatrixFactorization(n_users, n_items, n_factors=20, regularization=4)
        
        print("Training model...")
        # Train the model with early stopping
        training_errors, test_errors = model.fit_als(
            train_data, 
            n_iterations=15, 
            test_data=val_data,
            early_stopping=True,
            patience=2
        )
        
        # Evaluate on test set
        test_rmse = model.calculate_rmse(test_data)
        print(f"\nFinal Test RMSE: {test_rmse:.4f}")
    
    # Plot learning curves
    plot_learning_curves(training_errors, test_errors)

    # Generate recommendations for a sample user
    user_id = 50  # Example user ID
    if user_id >= n_users or user_id < 0:
        print(f"Invalid user_id: {user_id}. It must be between 0 and {n_users - 1}.")
        recommendations = None
    else:
        user_ratings = ratings[ratings['user_id'] == user_id]
        rated_items = user_ratings['item_id'].tolist()

        print(f"\nGenerating recommendations for user {user_id}...")
        recommendations = model.get_user_recommendations(user_id,
            n_recommendations=10, exclude_rated=True, rated_items=rated_items)

        # Print the recommendations
        if recommendations is not None and len(recommendations) > 0:
            print("\n Top 10 recommendations:")
            for i, (item_id, pred_rating) in enumerate(recommendations, 1):
                movie_title = movies.loc[movies['item_id'] == item_id, 'title'].values[0]
                print(f"{i}. {movie_title} (predicted rating: {pred_rating:.2f})")
        else:
            print(f"No recommendations available for user {user_id}.")
