
# import datetime
# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# import torch
# from werkzeug.security import generate_password_hash, check_password_hash
# import requests
# from math import radians, cos, sin, asin, sqrt
# from google.cloud import texttospeech_v1beta1 as texttospeech
# from google.cloud import translate_v2 as translate
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# import wikipedia
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# import re
# import nltk
# from transformers import BertModel, BertTokenizer
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from nltk.tokenize import sent_tokenize
# from flask import jsonify
# import logging
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('punkt')
# nltk.download('stopwords')
# from datetime import datetime, timezone
# from sklearn.cluster import KMeans



# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# db = SQLAlchemy(app)
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# GOOGLE_PLACES_API_KEY = 'AIzaSyC6JXdrY5SNL31rPWL1RUrln15ymEolLWQ'
# GOOGLE_APPLICATION_CREDENTIALS = 'C:/Users/DishaDiya/Downloads/final/code/tourismapp-428107-fe80d138c6d4.json'

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

# translate_client = translate.Client()

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     preferences = db.Column(db.String(500), nullable=True, default='')
#     activities = db.Column(db.String(500), nullable=True, default='')
#     ratings = db.Column(db.String(500), nullable=True, default='')

# # app.py (add this part to your existing code)
# class LikedPlace(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     place_name = db.Column(db.String(200), nullable=False)
#     place_vicinity = db.Column(db.String(200), nullable=False)
#     place_photo_url = db.Column(db.String(300), nullable=False)
#     place_rating = db.Column(db.String(10), nullable=True)

# class UserInteraction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'rate'
#     timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
#     rating = db.Column(db.Float, nullable=True)


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# def haversine(lon1, lat1, lon2, lat2):
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles
#     return c * r

# def get_place_details(lat, lng, search_radius=70000):
#     params = {
#         'location': f'{lat},{lng}',
#         'radius': search_radius,  # Search within the specified radius
#         'key': GOOGLE_PLACES_API_KEY,
#         'type': 'tourist_attraction'
#     }
#     response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', params=params)
#     return response.json().get('results', [])



# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         preferences = request.form.get('preferences')  # Get preferences
#         activities = request.form.get('activities')  # Get activities
#         ratings = request.form.get('ratings')  # Corrected field name here

#         hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
#         new_user = User(username=username, email=email, password=hashed_password,
#                         preferences=preferences, activities=activities, ratings=ratings)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')




# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password, password):
#             login_user(user)
#             return redirect(url_for('index'))
#         flash('Login failed. Check your email and password.', 'danger')
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


# @app.route('/account', methods=['GET', 'POST'])
# @login_required
# def account():
#     if request.method == 'POST':
#         # Update user information
#         current_user.username = request.form.get('username')
#         current_user.email = request.form.get('email')
        
#         # Update password if provided
#         new_password = request.form.get('new_password')
#         if new_password:
#             current_user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
        
#         # Update preferences, activities, and ratings
#         current_user.preferences = request.form.get('preferences', '')
#         current_user.activities = request.form.get('activities', '')
#         current_user.ratings = request.form.get('accommodation_rating', '')
        
#         try:
#             db.session.commit()
#             flash('Your account has been updated successfully!', 'success')
#         except Exception as e:
#             db.session.rollback()
#             flash(f'An error occurred: {str(e)}', 'danger')
        
#         return redirect(url_for('account'))
    
#     return render_template('account.html', user=current_user)



# def preprocess_user_data(user):
#     preferences = [pref.strip().lower() for pref in user.preferences.split(',')] if user.preferences else []
#     activities = [act.strip().lower() for act in user.activities.split(',')] if user.activities else []
#     ratings = [int(r) for r in user.ratings.split(',')] if user.ratings else []
#     return preferences, activities, ratings

# def get_user_interactions(user_id):
#     return UserInteraction.query.filter_by(user_id=user_id).all()

# # Function to get similar users
# def get_similar_users(user_id, n=5):
#     all_users = User.query.all()
#     user = User.query.get(user_id)
#     user_vector = set(user.preferences.split(',') + user.activities.split(','))
    
#     user_similarities = []
#     for other_user in all_users:
#         if other_user.id != user_id:
#             other_vector = set(other_user.preferences.split(',') + other_user.activities.split(','))
#             similarity = len(user_vector.intersection(other_vector)) / len(user_vector.union(other_vector))
#             user_similarities.append((other_user.id, similarity))
    
#     return sorted(user_similarities, key=lambda x: x[1], reverse=True)[:n]

# MIN_REVIEWS_THRESHOLD = 100  # Adjust this value as needed

# # Updated recommendation function
# # def get_recommendations(user, spots, num_recommendations=10):
# #     user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
# #     # Filter spots based on the minimum number of reviews
# #     famous_spots = [spot for spot in spots if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD]
    
# #     if not famous_spots:
# #         return []  # Return empty list if no famous spots are found

# #     # Content-based filtering
# #     place_descriptions = []
# #     for spot in famous_spots:
# #         description = (
# #             f"{spot.get('name', '')} "
# #             f"{spot.get('vicinity', '')} "
# #             f"{' '.join(spot.get('types', []))} "
# #             f"{spot.get('user_ratings_total', 0)} reviews "
# #             f"Rating: {spot.get('rating', 0)}"
# #         )
# #         place_descriptions.append(description)

# #     vectorizer = TfidfVectorizer(stop_words='english')
# #     tfidf_matrix = vectorizer.fit_transform(place_descriptions)

# #     user_profile = ' '.join(user_preferences + user_activities)
# #     user_tfidf = vectorizer.transform([user_profile])
    
# #     content_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

# #     # Collaborative filtering
# #     similar_users = get_similar_users(user.id)
# #     collaborative_scores = np.zeros(len(famous_spots))
# #     for similar_user_id, similarity in similar_users:
# #         similar_user_interactions = get_user_interactions(similar_user_id)
# #         for interaction in similar_user_interactions:
# #             spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
# #             if spot_index is not None:
# #                 if interaction.interaction_type == 'like':
# #                     collaborative_scores[spot_index] += similarity * 0.5
# #                 elif interaction.interaction_type == 'rate':
# #                     collaborative_scores[spot_index] += similarity * (interaction.rating / 5.0)

# #     # Combine scores
# #     final_scores = 0.6 * content_scores + 0.4 * collaborative_scores

# #     # Add diversity
# #     diversity_boost = np.random.uniform(0, 0.1, size=len(famous_spots))
# #     final_scores += diversity_boost

# #     # Time-based weighting for user interactions
# #     user_interactions = get_user_interactions(user.id)
# #     now = datetime.utcnow()
# #     for interaction in user_interactions:
# #         spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
# #         if spot_index is not None:
# #             time_diff = (now - interaction.timestamp).days
# #             time_weight = max(0, 1 - (time_diff / 30))  # Decay over 30 days
# #             if interaction.interaction_type == 'like':
# #                 final_scores[spot_index] += 0.3 * time_weight
# #             elif interaction.interaction_type == 'rate':
# #                 final_scores[spot_index] += 0.2 * (interaction.rating / 5.0) * time_weight

# #     # Rank and return top recommendations
# #     ranked_spots = sorted(zip(famous_spots, final_scores), key=lambda x: x[1], reverse=True)
# #     return ranked_spots[:num_recommendations]



# # def get_recommendations(user, spots, num_recommendations=10):
# #     user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
# #     # Filter spots based on the minimum number of reviews
# #     famous_spots = [spot for spot in spots if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD]
    
# #     if not famous_spots:
# #         return []  # Return empty list if no famous spots are found

# #     # Content-based filtering
# #     place_scores = []
# #     for spot in famous_spots:
# #         score = 0
# #         types = spot.get('types', [])
# #         name = spot.get('name', '').lower()
# #         vicinity = spot.get('vicinity', '').lower()

# #         # Check preferences
# #         for pref in user_preferences:
# #             if pref in types or pref in name or pref in vicinity:
# #                 score += 3  # Higher score for matching preferences

# #         # Check activities
# #         for activity in user_activities:
# #             if activity in types or activity in name or activity in vicinity:
# #                 score += 2  # Moderate score for matching activities

# #         # Check rating
# #         spot_rating = spot.get('rating', 0)
# #         if spot_rating:
# #             user_rating_preference = max(user_ratings) if user_ratings else 5
# #             score += (min(5, round(spot_rating)) / user_rating_preference) * 2

# #         # Additional score for diversity of matches
# #         unique_matches = len(set(pref for pref in user_preferences if pref in types or pref in name or pref in vicinity) |
# #                              set(activity for activity in user_activities if activity in types or activity in name or activity in vicinity))
# #         score += unique_matches * 0.5

# #         place_scores.append(score)

# #     # Normalize scores
# #     max_score = max(place_scores) if place_scores else 1
# #     normalized_scores = [score / max_score for score in place_scores]

# #     # Collaborative filtering (keep as is)
# #     similar_users = get_similar_users(user.id)
# #     collaborative_scores = np.zeros(len(famous_spots))
# #     for similar_user_id, similarity in similar_users:
# #         similar_user_interactions = get_user_interactions(similar_user_id)
# #         for interaction in similar_user_interactions:
# #             spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
# #             if spot_index is not None:
# #                 if interaction.interaction_type == 'like':
# #                     collaborative_scores[spot_index] += similarity * 0.5
# #                 elif interaction.interaction_type == 'rate':
# #                     collaborative_scores[spot_index] += similarity * (interaction.rating / 5.0)

# #     # Normalize collaborative scores
# #     max_collab_score = max(collaborative_scores) if collaborative_scores.any() else 1
# #     normalized_collab_scores = collaborative_scores / max_collab_score

# #     # Combine scores (give more weight to content-based filtering)
# #     final_scores = 0.7 * np.array(normalized_scores) + 0.3 * normalized_collab_scores

# #     # Add diversity
# #     diversity_boost = np.random.uniform(0, 0.1, size=len(famous_spots))
# #     final_scores += diversity_boost

# #     # Time-based weighting for user interactions (keep as is)
# #     user_interactions = get_user_interactions(user.id)
# #     now = datetime.utcnow()
# #     for interaction in user_interactions:
# #         spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
# #         if spot_index is not None:
# #             time_diff = (now - interaction.timestamp).days
# #             time_weight = max(0, 1 - (time_diff / 30))  # Decay over 30 days
# #             if interaction.interaction_type == 'like':
# #                 final_scores[spot_index] += 0.3 * time_weight
# #             elif interaction.interaction_type == 'rate':
# #                 final_scores[spot_index] += 0.2 * (interaction.rating / 5.0) * time_weight

# #     # Rank and return top recommendations
# #     ranked_spots = sorted(zip(famous_spots, final_scores), key=lambda x: x[1], reverse=True)
# #     return ranked_spots[:num_recommendations]

# # Update the recommend route




# def get_recommendations(user, spots, start_location, num_recommendations=10):
#     user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
#     # Filter spots based on the minimum number of reviews and 70km radius
#     famous_spots = [
#         spot for spot in spots 
#         if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD
#         and haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         ) <= 70
#     ]
    
#     if not famous_spots:
#         return []  # Return empty list if no famous spots are found

#     # Content-based filtering
#     place_scores = []
#     for spot in famous_spots:
#         score = 0
#         types = spot.get('types', [])
#         name = spot.get('name', '').lower()
#         vicinity = spot.get('vicinity', '').lower()

#         # Check preferences
#         for pref in user_preferences:
#             if pref in types or pref in name or pref in vicinity:
#                 score += 3  # Higher score for matching preferences

#         # Check activities
#         for activity in user_activities:
#             if activity in types or activity in name or activity in vicinity:
#                 score += 2  # Moderate score for matching activities

#         # Check rating
#         spot_rating = spot.get('rating', 0)
#         if spot_rating:
#             user_rating_preference = max(user_ratings) if user_ratings else 5
#             score += (min(5, round(spot_rating)) / user_rating_preference) * 2

#         # Additional score for diversity of matches
#         unique_matches = len(set(pref for pref in user_preferences if pref in types or pref in name or pref in vicinity) |
#                              set(activity for activity in user_activities if activity in types or activity in name or activity in vicinity))
#         score += unique_matches * 0.5

#         # Distance score (closer places get higher scores)
#         distance = haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         )
#         distance_score = 1 - (distance / 70)  # Normalize distance score
#         score += distance_score * 2  # Give significant weight to distance

#         place_scores.append(score)

#     # Normalize scores
#     max_score = max(place_scores) if place_scores else 1
#     normalized_scores = [score / max_score for score in place_scores]

#     # Collaborative filtering (keep as is)
#     similar_users = get_similar_users(user.id)
#     collaborative_scores = np.zeros(len(famous_spots))
#     for similar_user_id, similarity in similar_users:
#         similar_user_interactions = get_user_interactions(similar_user_id)
#         for interaction in similar_user_interactions:
#             spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
#             if spot_index is not None:
#                 if interaction.interaction_type == 'like':
#                     collaborative_scores[spot_index] += similarity * 0.5
#                 elif interaction.interaction_type == 'rate':
#                     collaborative_scores[spot_index] += similarity * (interaction.rating / 5.0)

#     # Normalize collaborative scores
#     max_collab_score = max(collaborative_scores) if collaborative_scores.any() else 1
#     normalized_collab_scores = collaborative_scores / max_collab_score

#     # Combine scores (give more weight to content-based filtering)
#     final_scores = 0.7 * np.array(normalized_scores) + 0.3 * normalized_collab_scores

#     # Add diversity
#     diversity_boost = np.random.uniform(0, 0.1, size=len(famous_spots))
#     final_scores += diversity_boost

#     # Time-based weighting for user interactions (keep as is)
#     user_interactions = get_user_interactions(user.id)
#     now = datetime.utcnow()
#     for interaction in user_interactions:
#         spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
#         if spot_index is not None:
#             time_diff = (now - interaction.timestamp).days
#             time_weight = max(0, 1 - (time_diff / 30))  # Decay over 30 days
#             if interaction.interaction_type == 'like':
#                 final_scores[spot_index] += 0.3 * time_weight
#             elif interaction.interaction_type == 'rate':
#                 final_scores[spot_index] += 0.2 * (interaction.rating / 5.0) * time_weight

#     # Rank and return top recommendations
#     ranked_spots = sorted(zip(famous_spots, final_scores), key=lambda x: x[1], reverse=True)
#     return ranked_spots[:num_recommendations]

# @app.route('/recommend', methods=['POST'])
# @login_required
# def recommend():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     start_location = {'lat': latitude, 'lng': longitude}

#     all_spots = get_place_details(latitude, longitude, search_radius=70000)  # 70km in meters
#     recommended_spots = get_recommendations(current_user, all_spots, start_location)

#     processed_spots = []
#     for spot, score in recommended_spots:
#         processed_spot = {
#             'name': spot['name'],
#             'vicinity': spot.get('vicinity', 'No address available'),
#             'rating': spot.get('rating', 'No rating'),
#             'photo_url': spot.get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400'),
#             'distance': haversine(latitude, longitude, spot['geometry']['location']['lat'], spot['geometry']['location']['lng'])
#         }
#         if processed_spot['photo_url'] != 'https://via.placeholder.com/400':
#             processed_spot['photo_url'] = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={processed_spot['photo_url']}&key={GOOGLE_PLACES_API_KEY}"
#         processed_spots.append(processed_spot)

#     return render_template('recommendations.html', spots=processed_spots)

# # Add a new route to record user interactions
# @app.route('/record_interaction', methods=['POST'])
# @login_required
# def record_interaction():
#     data = request.json
#     interaction = UserInteraction(
#         user_id=current_user.id,
#         place_id=data['place_id'],
#         interaction_type=data['interaction_type'],
#         rating=data.get('rating')
#     )
#     db.session.add(interaction)
#     db.session.commit()
#     return jsonify({'status': 'success'})



# logging.basicConfig(level=logging.INFO)
# @app.route('/like_place', methods=['POST'])
# @login_required
# def like_place():
#     if request.is_json:
#         data = request.get_json()
#         place_id = data.get('place_id')
#         place_name = data.get('place_name')
#         place_vicinity = data.get('place_vicinity')
#         place_photo_url = data.get('place_photo_url')
#         place_rating = data.get('place_rating')

#         existing_like = LikedPlace.query.filter_by(user_id=current_user.id, place_id=place_id).first()
#         if not existing_like:
#             try:
#                 liked_place = LikedPlace(
#                     user_id=current_user.id,
#                     place_id=place_id,
#                     place_name=place_name,
#                     place_vicinity=place_vicinity,
#                     place_photo_url=place_photo_url,
#                     place_rating=place_rating
#                 )
#                 db.session.add(liked_place)
#                 db.session.commit()
#                 return jsonify({'status': 'success', 'message': 'Place liked successfully!'})
#             except Exception as e:
#                 db.session.rollback()
#                 return jsonify({'status': 'error', 'message': f'Error saving like: {str(e)}'})
#         else:
#             return jsonify({'status': 'error', 'message': 'You already liked this place.'})
#     else:
#         return jsonify({'status': 'error', 'message': 'Unsupported Media Type'}), 415


# @app.route('/liked_places')
# @login_required
# def liked_places():
#     liked_places = LikedPlace.query.filter_by(user_id=current_user.id).all()
#     return render_template('liked_places.html', liked_places=liked_places)


# # @app.route('/like_place', methods=['POST'])
# # def like_place():
# #     if 'user_id' not in session:
# #         return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

# #     user_id = session['user_id']
# #     place_id = request.json.get('place_id')

# #     # Check if the place is already liked by the user
# #     existing_like = LikedPlace.query.filter_by(user_id=user_id, place_id=place_id).first()
# #     if existing_like:
# #         return jsonify({'status': 'error', 'message': 'Place already liked'}), 409

# #     # Add the liked place to the database
# #     new_like = LikedPlace(user_id=user_id, place_id=place_id)
# #     db.session.add(new_like)/
# #     db.session.commit()

# #     return jsonify({'status': 'success', 'message': 'Place liked successfully'}), 20



# def scrape_details(place_name):
#     try:
#         summary = wikipedia.summary(place_name)
#     except wikipedia.exceptions.DisambiguationError as e:
#         try:
#             summary = wikipedia.summary(e.options[0])
#         except Exception:
#             return None
#     except wikipedia.exceptions.PageError:
#         return None
#     except Exception:
#         return None

#     if summary:
#         return summarize_text(summary)
#     return None



# def get_sentence_embeddings(sentences):
#     # Load pre-trained BERT model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
    
#     # Tokenize sentences
#     inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Get the embeddings for the [CLS] token
#     embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
#     return embeddings

# def summarize_text(text, num_clusters=4):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)
    
#     # Check if the number of sentences is less than or equal to the number of clusters
#     if len(sentences) <= num_clusters:
#         return text  # Return the original text if it's too short
    
#     # Get embeddings for each sentence
#     embeddings = get_sentence_embeddings(sentences)
    
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(embeddings)
    
#     # Get cluster labels for each sentence
#     cluster_labels = kmeans.labels_
    
#     # Group sentences by their cluster
#     clustered_sentences = {i: [] for i in range(num_clusters)}
#     for sentence, label in zip(sentences, cluster_labels):
#         clustered_sentences[label].append(sentence)
    
#     # Select the first sentence from each cluster to form the summary
#     summary_sentences = [clustered_sentences[i][0] for i in range(num_clusters)]
    
#     # Join the summary sentences into a final summary
#     summary = ' '.join(summary_sentences)
    
#     return summary


# def visualize_embeddings(embeddings):
#     # Normalize the embeddings to the range [0, 1]
#     min_val = np.min(embeddings)
#     max_val = np.max(embeddings)
#     normalized_embeddings = (embeddings - min_val) / (max_val - min_val)

#     # Define ASCII characters for visualization
#     chars = np.array([' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'])
    
#     # Map normalized values to ASCII characters
#     indices = (normalized_embeddings * (len(chars) - 1)).astype(int)
#     ascii_matrix = chars[indices]

#     # Print the ASCII representation of the embeddings
#     for row in ascii_matrix:
#         print(''.join(row))

# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # def summarize_text(text, num_sentences=3):
# #     sentences = sent_tokenize(text)
# #     if len(sentences) <= num_sentences:
# #         return text

# #     # Get sentence embeddings
# #     embeddings = model.encode(sentences, convert_to_tensor=True)

# #     # Compute cosine similarity matrix
# #     cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

# #     # Rank sentences based on their similarity to the entire document
# #     sentence_scores = cosine_sim_matrix.sum(axis=1)

# #     # Get the top N sentences
# #     top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
# #     top_sentences = [sentences[idx] for idx in sorted(top_sentence_indices)]

# #     # Join the top sentences into a summary
# #     summary = ' '.join(top_sentences)
# #     return summary



# @app.route('/find_spots', methods=['POST'])
# @login_required
# def find_spots():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     language = request.form['language']
#     user_radius = request.form.get('user_radius')
#     custom_user_radius = request.form.get('custom_user_radius')

#     if user_radius == 'custom':
#         user_radius = int(custom_user_radius)
#     else:
#         user_radius = int(user_radius)

#     # Fetch places within a large radius (to ensure enough results)
#     places = get_place_details(latitude, longitude)

#     # Filter out places without essential data and those with insufficient ratings
#     filtered_places = [
#         place for place in places
#         if 'name' in place and 'geometry' in place and 'location' in place['geometry'] and place.get('user_ratings_total', 0) > 200
#     ]

#     # Calculate distance for each place from the user's location
#     places_with_distance = [
#         (place, haversine(longitude, latitude, place['geometry']['location']['lng'], place['geometry']['location']['lat']))
#         for place in filtered_places
#     ]

#     # Filter places based on the user-specified radius
#     within_radius_places = [place for place in places_with_distance if place[1] <= user_radius]

#     # Sort places by distance
#     within_radius_places.sort(key=lambda x: x[1])

#     # Select top spots within the radius
#     top_spots = within_radius_places[:10]

#     # Initialize translation client
#     translated_spots = []
#     client = texttospeech.TextToSpeechClient()
#     for spot, distance in top_spots:
#         place_id = spot['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         # Translate place details
#         name_translation = translate_client.translate(spot['name'], target_language=language)
#         vicinity_translation = translate_client.translate(spot.get('vicinity', ''), target_language=language)
        
#         # Scrape and summarize description, if available
#         description = scrape_details(spot['name'])
#         if not description:
#             description = details.get('editorial_summary', {}).get('overview', 'No detailed description available.')
#         else:
#             description = summarize_text(description)
#         description_translation = translate_client.translate(description, target_language=language)

#         # Truncate the translated description to fit the limit
#         truncated_description = truncate_text(description_translation['translatedText'], 5000)

#         # Handle photos
#         photo_reference = None
#         if 'photos' in spot:
#             photo_reference = spot['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         # Get rating and reviews
#         rating = details.get('rating', 'No rating')
#         reviews = details.get('reviews', [])

#         # Append translated spot to the list
#         translated_spots.append({
#             'name': name_translation['translatedText'],
#             'vicinity': vicinity_translation['translatedText'],
#             'description': truncated_description,
#             'distance': distance,
#             'photo_url': photo_url,
#             'rating': rating,
#             'reviews': reviews
#         })

#         # Generate audio for the description
#         text = truncated_description
#         synthesis_input = texttospeech.SynthesisInput(text=text)
#         voice = texttospeech.VoiceSelectionParams(language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
#         audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#         tts_response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

#         # Save audio file
#         audio_dir = "C:/Users/DishaDiya/Downloads/final/code/static/audio/"
#         os.makedirs(audio_dir, exist_ok=True)
#         audio_filename = f"{audio_dir}{spot['name'].replace(' ', '_')}_description.mp3"

#         with open(audio_filename, "wb") as out:
#             out.write(tts_response.audio_content)

#         # Add audio file link to the spot data
#         translated_spots[-1]['audio'] = f"audio/{spot['name'].replace(' ', '_')}_description.mp3"

#     # Get nearby cafes and restaurants within the user-specified radius
#     location = {'lat': latitude, 'lng': longitude}
#     cafes = get_nearby_places(location, 'cafe',70000)
#     restaurants = get_nearby_places(location, 'restaurant', 70000)

#     # Get start location details
#     start_location_details = get_place_details(latitude, longitude, search_radius=50)
#     start_location = {
#         'name': start_location_details[0].get('name', 'Unknown Location') if start_location_details else 'Unknown Location',
#         'vicinity': start_location_details[0].get('vicinity', '') if start_location_details else '',
#         'photo_url': start_location_details[0].get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400') if start_location_details else 'https://via.placeholder.com/400',
#         'rating': start_location_details[0].get('rating', 'No rating') if start_location_details else 'No rating'
#     }

#     return render_template('results.html', spots=translated_spots, cafes=cafes, restaurants=restaurants, start_location=start_location)


# def truncate_text(text, max_bytes):
#     """Truncate text to ensure it does not exceed the specified number of bytes."""
#     encoded_text = text.encode('utf-8')
#     if len(encoded_text) <= max_bytes:
#         return text

#     truncated_text = encoded_text[:max_bytes].decode('utf-8', errors='ignore')
#     return truncated_text



# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places

# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, host='127.0.0.1', port=5003)


















# import datetime
# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# import torch
# from werkzeug.security import generate_password_hash, check_password_hash
# import requests
# from math import radians, cos, sin, asin, sqrt
# from google.cloud import texttospeech_v1beta1 as texttospeech
# from google.cloud import translate_v2 as translate
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# import wikipedia
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# import re
# import nltk
# from transformers import BertModel, BertTokenizer
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from nltk.tokenize import sent_tokenize
# from flask import jsonify
# import logging
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('punkt')
# nltk.download('stopwords')
# from datetime import datetime, timezone
# from sklearn.cluster import KMeans



# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# db = SQLAlchemy(app)
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# GOOGLE_PLACES_API_KEY = 'AIzaSyC6JXdrY5SNL31rPWL1RUrln15ymEolLWQ'
# GOOGLE_APPLICATION_CREDENTIALS = 'C:/Users/DishaDiya/Downloads/final/code/tourismapp-428107-fe80d138c6d4.json'

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

# translate_client = translate.Client()

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     preferences = db.Column(db.String(500), nullable=True, default='')
#     activities = db.Column(db.String(500), nullable=True, default='')
#     ratings = db.Column(db.String(500), nullable=True, default='')

# class LikedPlace(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     place_name = db.Column(db.String(200), nullable=False)
#     place_vicinity = db.Column(db.String(300))
#     place_rating = db.Column(db.Float)
#     place_photo_url = db.Column(db.String(500))
#     timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))

#     user = db.relationship('User', backref=db.backref('liked_places', lazy=True))

# @app.route('/like_place', methods=['POST'])
# @login_required
# def like_place():
#     place_id = request.form.get('place_id')
#     place_name = request.form.get('place_name')
#     place_vicinity = request.form.get('place_vicinity')
#     place_rating = request.form.get('place_rating')
#     place_photo_url = request.form.get('place_photo_url')

#     existing_like = LikedPlace.query.filter_by(user_id=current_user.id, place_id=place_id).first()

#     if existing_like:
#         db.session.delete(existing_like)
#         db.session.commit()
#         return jsonify({'status': 'unliked'})
#     else:
#         new_like = LikedPlace(
#             user_id=current_user.id,
#             place_id=place_id,
#             place_name=place_name,
#             place_vicinity=place_vicinity,
#             place_rating=place_rating,
#             place_photo_url=place_photo_url
#         )
#         db.session.add(new_like)
#         db.session.commit()
#         return jsonify({'status': 'liked'})

# @app.route('/liked_place')
# @login_required
# def liked_places():
#     liked_places = LikedPlace.query.filter_by(user_id=current_user.id).order_by(LikedPlace.timestamp.desc()).all()
#     return render_template('liked_places.html', liked_places=liked_places)


# class UserInteraction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'rate'
#     timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
#     rating = db.Column(db.Float, nullable=True)


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# def haversine(lon1, lat1, lon2, lat2):
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles
#     return c * r

# def get_place_details(lat, lng, search_radius=70000):
#     params = {
#         'location': f'{lat},{lng}',
#         'radius': search_radius,  # Search within the specified radius
#         'key': GOOGLE_PLACES_API_KEY,
#         'type': 'tourist_attraction'
#     }
#     response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', params=params)
#     return response.json().get('results', [])



# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         preferences = request.form.get('preferences')  # Get preferences
#         activities = request.form.get('activities')  # Get activities
#         ratings = request.form.get('ratings')  # Corrected field name here

#         hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
#         new_user = User(username=username, email=email, password=hashed_password,
#                         preferences=preferences, activities=activities, ratings=ratings)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')




# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password, password):
#             login_user(user)
#             return redirect(url_for('index'))
#         flash('Login failed. Check your email and password.', 'danger')
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


# @app.route('/account', methods=['GET', 'POST'])
# @login_required
# def account():
#     if request.method == 'POST':
#         if 'username' in request.form:
#             # Updating personal information
#             current_user.username = request.form['username']
#             current_user.email = request.form['email']
#             if request.form['new_password']:
#                 current_user.set_password(request.form['new_password'])
        
#         # Always include username and email when updating
#         username = current_user.username
#         email = current_user.email
        
#         if 'preferences' in request.form:
#             current_user.preferences = request.form['preferences']
        
#         if 'activities' in request.form:
#             current_user.activities = request.form['activities']
        
#         if 'accommodation_rating' in request.form:
#             current_user.ratings = request.form['accommodation_rating']
        
#         # Update the user in the database
#         db.session.commit()
        
#         flash('Your account has been updated!', 'success')
#         return redirect(url_for('account'))
    
#     return render_template('account.html', title='Account')



# def preprocess_user_data(user):
#     preferences = [pref.strip().lower() for pref in user.preferences.split(',')] if user.preferences else []
#     activities = [act.strip().lower() for act in user.activities.split(',')] if user.activities else []
#     ratings = [int(r) for r in user.ratings.split(',')] if user.ratings else []
#     return preferences, activities, ratings

# def get_user_interactions(user_id):
#     return UserInteraction.query.filter_by(user_id=user_id).all()

# # Function to get similar users
# def get_similar_users(user_id, n=5):
#     all_users = User.query.all()
#     user = User.query.get(user_id)
#     user_vector = set(user.preferences.split(',') + user.activities.split(','))
    
#     user_similarities = []
#     for other_user in all_users:
#         if other_user.id != user_id:
#             other_vector = set(other_user.preferences.split(',') + other_user.activities.split(','))
#             similarity = len(user_vector.intersection(other_vector)) / len(user_vector.union(other_vector))
#             user_similarities.append((other_user.id, similarity))
    
#     return sorted(user_similarities, key=lambda x: x[1], reverse=True)[:n]

# MIN_REVIEWS_THRESHOLD = 100  # Adjust this value as needed

# def get_recommendations(user, spots, start_location, num_recommendations=10):
#     user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
#     # Filter spots based on the minimum number of reviews and 70km radius
#     famous_spots = [
#         spot for spot in spots 
#         if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD
#         and haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         ) <= 70
#     ]
    
#     if not famous_spots:
#         return []  # Return empty list if no famous spots are found

#     # Content-based filtering
#     place_scores = []
#     for spot in famous_spots:
#         score = 0
#         types = spot.get('types', [])
#         name = spot.get('name', '').lower()
#         vicinity = spot.get('vicinity', '').lower()

#         # Check preferences
#         for pref in user_preferences:
#             if pref in types or pref in name or pref in vicinity:
#                 score += 3  # Higher score for matching preferences

#         # Check activities
#         for activity in user_activities:
#             if activity in types or activity in name or activity in vicinity:
#                 score += 2  # Moderate score for matching activities

#         # Check rating
#         spot_rating = spot.get('rating', 0)
#         if spot_rating:
#             user_rating_preference = max(user_ratings) if user_ratings else 5
#             score += (min(5, round(spot_rating)) / user_rating_preference) * 2

#         # Additional score for diversity of matches
#         unique_matches = len(set(pref for pref in user_preferences if pref in types or pref in name or pref in vicinity) |
#                              set(activity for activity in user_activities if activity in types or activity in name or activity in vicinity))
#         score += unique_matches * 0.5

#         # Distance score (closer places get higher scores)
#         distance = haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         )
#         distance_score = 1 - (distance / 70)  # Normalize distance score
#         score += distance_score * 2  # Give significant weight to distance

#         place_scores.append(score)

#     # Normalize scores
#     max_score = max(place_scores) if place_scores else 1
#     normalized_scores = [score / max_score for score in place_scores]

#     # Collaborative filtering (keep as is)
#     similar_users = get_similar_users(user.id)
#     collaborative_scores = np.zeros(len(famous_spots))
#     for similar_user_id, similarity in similar_users:
#         similar_user_interactions = get_user_interactions(similar_user_id)
#         for interaction in similar_user_interactions:
#             spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
#             if spot_index is not None:
#                 if interaction.interaction_type == 'like':
#                     collaborative_scores[spot_index] += similarity * 0.5
#                 elif interaction.interaction_type == 'rate':
#                     collaborative_scores[spot_index] += similarity * (interaction.rating / 5.0)

#     # Normalize collaborative scores
#     max_collab_score = max(collaborative_scores) if collaborative_scores.any() else 1
#     normalized_collab_scores = collaborative_scores / max_collab_score

#     # Combine scores (give more weight to content-based filtering)
#     final_scores = 0.7 * np.array(normalized_scores) + 0.3 * normalized_collab_scores

#     # Add diversity
#     diversity_boost = np.random.uniform(0, 0.1, size=len(famous_spots))
#     final_scores += diversity_boost

#     # Time-based weighting for user interactions (keep as is)
#     user_interactions = get_user_interactions(user.id)
#     now = datetime.utcnow()
#     for interaction in user_interactions:
#         spot_index = next((i for i, spot in enumerate(famous_spots) if spot['place_id'] == interaction.place_id), None)
#         if spot_index is not None:
#             time_diff = (now - interaction.timestamp).days
#             time_weight = max(0, 1 - (time_diff / 30))  # Decay over 30 days
#             if interaction.interaction_type == 'like':
#                 final_scores[spot_index] += 0.3 * time_weight
#             elif interaction.interaction_type == 'rate':
#                 final_scores[spot_index] += 0.2 * (interaction.rating / 5.0) * time_weight

#     # Rank and return top recommendations
#     ranked_spots = sorted(zip(famous_spots, final_scores), key=lambda x: x[1], reverse=True)
#     return ranked_spots[:num_recommendations]

# @app.route('/recommend', methods=['POST'])
# @login_required
# def recommend():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     start_location = {'lat': latitude, 'lng': longitude}

#     all_spots = get_place_details(latitude, longitude, search_radius=70000)  # 70km in meters
#     recommended_spots = get_recommendations(current_user, all_spots, start_location)

#     processed_spots = []
#     for spot, score in recommended_spots:
#         processed_spot = {
#             'name': spot['name'],
#             'vicinity': spot.get('vicinity', 'No address available'),
#             'rating': spot.get('rating', 'No rating'),
#             'photo_url': spot.get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400'),
#             'distance': haversine(latitude, longitude, spot['geometry']['location']['lat'], spot['geometry']['location']['lng'])
#         }
#         if processed_spot['photo_url'] != 'https://via.placeholder.com/400':
#             processed_spot['photo_url'] = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={processed_spot['photo_url']}&key={GOOGLE_PLACES_API_KEY}"
#         processed_spots.append(processed_spot)

#     return render_template('recommendations.html', spots=processed_spots)

# # Add a new route to record user interactions
# @app.route('/record_interaction', methods=['POST'])
# @login_required
# def record_interaction():
#     data = request.json
#     interaction = UserInteraction(
#         user_id=current_user.id,
#         place_id=data['place_id'],
#         interaction_type=data['interaction_type'],
#         rating=data.get('rating')
#     )
#     db.session.add(interaction)
#     db.session.commit()
#     return jsonify({'status': 'success'})



# logging.basicConfig(level=logging.INFO)
# def scrape_details(place_name):
#     try:
#         summary = wikipedia.summary(place_name)
#     except wikipedia.exceptions.DisambiguationError as e:
#         try:
#             summary = wikipedia.summary(e.options[0])
#         except Exception:
#             return None
#     except wikipedia.exceptions.PageError:
#         return None
#     except Exception:
#         return None

#     if summary:
#         return summarize_text(summary)
#     return None



# def get_sentence_embeddings(sentences):
#     # Load pre-trained BERT model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
    
#     # Tokenize sentences
#     inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Get the embeddings for the [CLS] token
#     embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
#     return embeddings

# def summarize_text(text, num_clusters=4):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)
    
#     # Check if the number of sentences is less than or equal to the number of clusters
#     if len(sentences) <= num_clusters:
#         return text  # Return the original text if it's too short
    
#     # Get embeddings for each sentence
#     embeddings = get_sentence_embeddings(sentences)
    
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(embeddings)
    
#     # Get cluster labels for each sentence
#     cluster_labels = kmeans.labels_
    
#     # Group sentences by their cluster
#     clustered_sentences = {i: [] for i in range(num_clusters)}
#     for sentence, label in zip(sentences, cluster_labels):
#         clustered_sentences[label].append(sentence)
    
#     # Select the first sentence from each cluster to form the summary
#     summary_sentences = [clustered_sentences[i][0] for i in range(num_clusters)]
    
#     # Join the summary sentences into a final summary
#     summary = ' '.join(summary_sentences)
    
#     return summary


# def visualize_embeddings(embeddings):
#     # Normalize the embeddings to the range [0, 1]
#     min_val = np.min(embeddings)
#     max_val = np.max(embeddings)
#     normalized_embeddings = (embeddings - min_val) / (max_val - min_val)

#     # Define ASCII characters for visualization
#     chars = np.array([' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'])
    
#     # Map normalized values to ASCII characters
#     indices = (normalized_embeddings * (len(chars) - 1)).astype(int)
#     ascii_matrix = chars[indices]

#     # Print the ASCII representation of the embeddings
#     for row in ascii_matrix:
#         print(''.join(row))

# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # def summarize_text(text, num_sentences=3):
# #     sentences = sent_tokenize(text)
# #     if len(sentences) <= num_sentences:
# #         return text

# #     # Get sentence embeddings
# #     embeddings = model.encode(sentences, convert_to_tensor=True)

# #     # Compute cosine similarity matrix
# #     cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

# #     # Rank sentences based on their similarity to the entire document
# #     sentence_scores = cosine_sim_matrix.sum(axis=1)

# #     # Get the top N sentences
# #     top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
# #     top_sentences = [sentences[idx] for idx in sorted(top_sentence_indices)]

# #     # Join the top sentences into a summary
# #     summary = ' '.join(top_sentences)
# #     return summary



# @app.route('/find_spots', methods=['POST'])
# @login_required
# def find_spots():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     language = request.form['language']
#     user_radius = request.form.get('user_radius')
#     custom_user_radius = request.form.get('custom_user_radius')

#     if user_radius == 'custom':
#         user_radius = int(custom_user_radius)
#     else:
#         user_radius = int(user_radius)

#     # Fetch places within a large radius (to ensure enough results)
#     places = get_place_details(latitude, longitude)

#     # Filter out places without essential data and those with insufficient ratings
#     filtered_places = [
#         place for place in places
#         if 'name' in place and 'geometry' in place and 'location' in place['geometry'] and place.get('user_ratings_total', 0) > 200
#     ]

#     # Calculate distance for each place from the user's location
#     places_with_distance = [
#         (place, haversine(longitude, latitude, place['geometry']['location']['lng'], place['geometry']['location']['lat']))
#         for place in filtered_places
#     ]

#     # Filter places based on the user-specified radius
#     within_radius_places = [place for place in places_with_distance if place[1] <= user_radius]

#     # Sort places by distance
#     within_radius_places.sort(key=lambda x: x[1])

#     # Select top spots within the radius
#     top_spots = within_radius_places[:10]

#     # Initialize translation client
#     translated_spots = []
#     client = texttospeech.TextToSpeechClient()
#     for spot, distance in top_spots:
#         place_id = spot['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         # Translate place details
#         name_translation = translate_client.translate(spot['name'], target_language=language)
#         vicinity_translation = translate_client.translate(spot.get('vicinity', ''), target_language=language)
        
#         # Scrape and summarize description, if available
#         description = scrape_details(spot['name'])
#         if not description:
#             description = details.get('editorial_summary', {}).get('overview', 'No detailed description available.')
#         else:
#             description = summarize_text(description)
#         description_translation = translate_client.translate(description, target_language=language)

#         # Truncate the translated description to fit the limit
#         truncated_description = truncate_text(description_translation['translatedText'], 5000)

#         # Handle photos
#         photo_reference = None
#         if 'photos' in spot:
#             photo_reference = spot['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         # Get rating and reviews
#         rating = details.get('rating', 'No rating')
#         reviews = details.get('reviews', [])

#         # Append translated spot to the list
#         translated_spots.append({
#             'name': name_translation['translatedText'],
#             'vicinity': vicinity_translation['translatedText'],
#             'description': truncated_description,
#             'distance': distance,
#             'photo_url': photo_url,
#             'rating': rating,
#             'reviews': reviews
#         })

#         # Generate audio for the description
#         text = truncated_description
#         synthesis_input = texttospeech.SynthesisInput(text=text)
#         voice = texttospeech.VoiceSelectionParams(language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
#         audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#         tts_response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

#         # Save audio file
#         audio_dir = "C:/Users/DishaDiya/Downloads/final/code/static/audio/"
#         os.makedirs(audio_dir, exist_ok=True)
#         audio_filename = f"{audio_dir}{spot['name'].replace(' ', '_')}_description.mp3"

#         with open(audio_filename, "wb") as out:
#             out.write(tts_response.audio_content)

#         # Add audio file link to the spot data
#         translated_spots[-1]['audio'] = f"audio/{spot['name'].replace(' ', '_')}_description.mp3"

#     # Get nearby cafes and restaurants within the user-specified radius
#     location = {'lat': latitude, 'lng': longitude}
#     cafes = get_nearby_places(location, 'cafe',70000)
#     restaurants = get_nearby_places(location, 'restaurant', 70000)

#     # Get start location details
#     start_location_details = get_place_details(latitude, longitude, search_radius=50)
#     start_location = {
#         'name': start_location_details[0].get('name', 'Unknown Location') if start_location_details else 'Unknown Location',
#         'vicinity': start_location_details[0].get('vicinity', '') if start_location_details else '',
#         'photo_url': start_location_details[0].get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400') if start_location_details else 'https://via.placeholder.com/400',
#         'rating': start_location_details[0].get('rating', 'No rating') if start_location_details else 'No rating'
#     }

#     return render_template('results.html', spots=translated_spots, cafes=cafes, restaurants=restaurants, start_location=start_location)


# def truncate_text(text, max_bytes):
#     """Truncate text to ensure it does not exceed the specified number of bytes."""
#     encoded_text = text.encode('utf-8')
#     if len(encoded_text) <= max_bytes:
#         return text

#     truncated_text = encoded_text[:max_bytes].decode('utf-8', errors='ignore')
#     return truncated_text



# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places

# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places


# @app.route('/about')
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, host='127.0.0.1', port=5003)


































# import datetime
# import os
# from sqlite3 import IntegrityError
# from flask import Flask, render_template, request, redirect, url_for, flash
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# import torch
# from werkzeug.security import generate_password_hash, check_password_hash
# import requests
# from math import radians, cos, sin, asin, sqrt
# from google.cloud import texttospeech_v1beta1 as texttospeech
# from google.cloud import translate_v2 as translate
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# import wikipedia
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# import re
# import nltk
# from transformers import BertModel, BertTokenizer
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from nltk.tokenize import sent_tokenize
# from flask import jsonify
# import logging
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('punkt')
# nltk.download('stopwords')
# from datetime import datetime, timezone
# from sklearn.cluster import KMeans



# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# db = SQLAlchemy(app)
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# GOOGLE_PLACES_API_KEY = 'AIzaSyC6JXdrY5SNL31rPWL1RUrln15ymEolLWQ'
# GOOGLE_APPLICATION_CREDENTIALS = 'C:/Users/DishaDiya/Downloads/final/code/tourismapp-428107-fe80d138c6d4.json'

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

# translate_client = translate.Client()

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     preferences = db.Column(db.String(500), nullable=True, default='')
#     activities = db.Column(db.String(500), nullable=True, default='')
#     ratings = db.Column(db.String(500), nullable=True, default='')

# class LikedPlace(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     place_name = db.Column(db.String(200), nullable=False)
#     place_vicinity = db.Column(db.String(300))
#     place_rating = db.Column(db.Float)
#     place_photo_url = db.Column(db.String(500))
#     timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))

#     user = db.relationship('User', backref=db.backref('liked_places', lazy=True))

#     __table_args__ = (db.UniqueConstraint('user_id', 'place_id', name='_user_place_uc'),)

# from flask import session

# @app.route('/like_place', methods=['POST'])
# @login_required
# def like_place():
#     place_id = request.form.get('place_id')
#     place_name = request.form.get('place_name')
#     place_vicinity = request.form.get('place_vicinity')
#     place_rating = request.form.get('place_rating')
#     place_photo_url = request.form.get('place_photo_url')

#     existing_like = LikedPlace.query.filter_by(user_id=current_user.id, place_id=place_id).first()

#     if existing_like:
#         db.session.delete(existing_like)
#         db.session.commit()
#         if 'liked_places' in session:
#             session['liked_places'] = [place for place in session['liked_places'] if place['place_id'] != place_id]
#             session.modified = True
#         return jsonify({'status': 'unliked'})
#     else:
#         new_like = LikedPlace(
#             user_id=current_user.id,
#             place_id=place_id,
#             place_name=place_name,
#             place_vicinity=place_vicinity,
#             place_rating=place_rating,
#             place_photo_url=place_photo_url,
#             timestamp=datetime.now(timezone.utc)
#         )
#         try:
#             db.session.add(new_like)
#             db.session.commit()
            
#             if 'liked_places' not in session:
#                 session['liked_places'] = []
            
#             session['liked_places'].append({
#                 'place_id': place_id,
#                 'place_name': place_name,
#                 'place_vicinity': place_vicinity,
#                 'place_rating': place_rating,
#                 'place_photo_url': place_photo_url
#             })
#             session.modified = True
            
#             return jsonify({'status': 'liked'})
#         except IntegrityError:
#             db.session.rollback()
#             return jsonify({'status': 'error', 'message': 'This place is already liked'}), 400
#         except Exception as e:
#             db.session.rollback()
#             return jsonify({'status': 'error', 'message': str(e)}), 400

# @app.route('/liked_places')
# @login_required
# def liked_places():
#     liked_places = LikedPlace.query.filter_by(user_id=current_user.id).order_by(LikedPlace.timestamp.desc()).all()
#     return render_template('liked_places.html', liked_places=liked_places)



# class UserInteraction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     place_id = db.Column(db.String(100), nullable=False)
#     interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'rate'
#     timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
#     rating = db.Column(db.Float, nullable=True)


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# def haversine(lon1, lat1, lon2, lat2):
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles
#     return c * r

# def get_place_details(lat, lng, search_radius=70000):
#     params = {
#         'location': f'{lat},{lng}',
#         'radius': search_radius,
#         'key': GOOGLE_PLACES_API_KEY,
#         'type': 'point_of_interest',  # This is more inclusive than just 'tourist_attraction'
#         'fields': 'place_id,name,types,vicinity,rating,user_ratings_total,geometry,photos'
#     }
#     response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', params=params)
#     return response.json().get('results', [])





# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         preferences = request.form.get('preferences')  # Get preferences
#         activities = request.form.get('activities')  # Get activities
#         ratings = request.form.get('ratings')  # Corrected field name here

#         hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
#         new_user = User(username=username, email=email, password=hashed_password,
#                         preferences=preferences, activities=activities, ratings=ratings)
#         db.session.add(new_user)
#         db.session.commit()
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('register.html')




# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password, password):
#             login_user(user)
#             return redirect(url_for('index'))
#         flash('Login failed. Check your email and password.', 'danger')
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


# @app.route('/account', methods=['GET', 'POST'])
# @login_required
# def account():
#     if request.method == 'POST':
#         if 'username' in request.form:
#             # Updating personal information
#             current_user.username = request.form['username']
#             current_user.email = request.form['email']
#             if request.form['new_password']:
#                 current_user.set_password(request.form['new_password'])
        
#         # Always include username and email when updating
#         username = current_user.username
#         email = current_user.email
        
#         if 'preferences' in request.form:
#             current_user.preferences = request.form['preferences']
        
#         if 'activities' in request.form:
#             current_user.activities = request.form['activities']
        
#         if 'accommodation_rating' in request.form:
#             current_user.ratings = request.form['accommodation_rating']
        
#         # Update the user in the database
#         db.session.commit()
        
#         flash('Your account has been updated!', 'success')
#         return redirect(url_for('account'))
    
#     return render_template('account.html', title='Account')



# def preprocess_user_data(user):
#     preference_mapping = {
#         'amusement park': ['amusement_park', 'theme_park', 'water_park'],
#         'beach': ['beach', 'coast', 'seaside', 'ocean'],
#         'mountain': ['mountain', 'hill', 'peak', 'alpine'],
#         'city': ['city', 'urban', 'metropolis'],
#         'historical': ['historical', 'ancient', 'heritage', 'museum'],
#         'nature': ['nature', 'park', 'forest', 'wildlife'],
#         # Add more mappings as needed
#     }
    
#     raw_preferences = [pref.strip().lower() for pref in user.preferences.split(',')] if user.preferences else []
#     preferences = []
#     for pref in raw_preferences:
#         preferences.extend(preference_mapping.get(pref, [pref]))
    
#     activities = [act.strip().lower() for act in user.activities.split(',')] if user.activities else []
#     ratings = [int(r) for r in user.ratings.split(',')] if user.ratings else []
#     return preferences, activities, ratings
#     return preferences, activities, ratings

# MIN_REVIEWS_THRESHOLD= 100

# def get_recommendations(user, spots, start_location, num_recommendations=10, displayed_spots=None):
#     if displayed_spots is None:
#         displayed_spots = set()
#     else:
#         displayed_spots = set(spot['place_id'] for spot in displayed_spots)

#     user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
#     translated_spots = set(session.get('translated_spots', []))
#     # Filter spots based on the minimum number of reviews, 70km radius, and not already displayed
#     famous_spots = [
#         spot for spot in spots 
#         if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD
#         and haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         ) <= 70
#         and spot['place_id'] not in displayed_spots
#     ]
    
#     if not famous_spots:
#         return []  # Return empty list if no famous spots are found

#     return content_based_filtering(user_preferences, user_activities, user_ratings, famous_spots, start_location, num_recommendations)




# def content_based_filtering(user_preferences, user_activities, user_ratings, famous_spots, start_location, num_recommendations):
#     place_scores = []
#     for spot in famous_spots:
#         score = 0
#         types = spot.get('types', [])
#         name = spot.get('name', '').lower()
#         vicinity = spot.get('vicinity', '').lower()

#         # Check for preference matches
#         preference_match = any(pref in types or pref in name or pref in vicinity for pref in user_preferences)
        
#         # Consider all spots, but give a significant boost to those matching preferences
#         if preference_match:
#             score += 50  # Significant boost for matching preferences

#         # Check activities
#         for activity in user_activities:
#             if activity in types or activity in name or activity in vicinity:
#                 score += 5  # Moderate score for matching activities

#         # Check rating
#         spot_rating = spot.get('rating', 0)
#         if spot_rating:
#             user_rating_preference = max(user_ratings) if user_ratings else 5
#             score += (min(5, round(spot_rating)) / user_rating_preference) * 3

#         # Distance score (closer places get higher scores)
#         distance = haversine(
#             start_location['lat'], start_location['lng'],
#             spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
#         )
#         distance_score = 1 - (distance / 70)  # Normalize distance score
#         score += distance_score * 10  # Give significant weight to distance

#         place_scores.append((spot, score))

#     # Rank and return top recommendations
#     ranked_spots = sorted(place_scores, key=lambda x: x[1], reverse=True)
#     return ranked_spots[:num_recommendations]

#     # Rank and return top recommendations
#     ranked_spots = sorted(place_scores, key=lambda x: x[1], reverse=True)
#     return ranked_spots[:num_recommendations]



# @app.route('/recommend', methods=['POST'])
# @login_required
# def recommend():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     start_location = {'lat': latitude, 'lng': longitude}

#     # Get the displayed spots from the session
#     displayed_spots = session.get('displayed_spots', [])

#     # Get the translated spots from the session
#     translated_spots = set(session.get('translated_spots', []))

#     all_spots = get_place_details(latitude, longitude, search_radius=70000)  # 70km in meters
    
#     # Log user preferences and all spots for debugging
#     app.logger.info(f"User preferences: {current_user.preferences}")
#     app.logger.info(f"All spots: {[spot['name'] for spot in all_spots]}")

#     recommended_spots = get_recommendations(current_user, all_spots, start_location, displayed_spots=displayed_spots)

#     # Log recommended spots for debugging
#     app.logger.info(f"Recommended spots: {[spot[0]['name'] for spot in recommended_spots]}")

#     processed_spots = []
#     for spot, score in recommended_spots:
#         if spot['place_id'] not in translated_spots:
#             processed_spot = {
#                 'place_id': spot['place_id'],
#                 'name': spot['name'],
#                 'vicinity': spot.get('vicinity', 'No address available'),
#                 'rating': spot.get('rating', 'No rating'),
#                 'photo_url': spot.get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400'),
#                 'distance': haversine(latitude, longitude, spot['geometry']['location']['lat'], spot['geometry']['location']['lng'])
#             }
#             if processed_spot['photo_url'] != 'https://via.placeholder.com/400':
#                 processed_spot['photo_url'] = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={processed_spot['photo_url']}&key={GOOGLE_PLACES_API_KEY}"
#             processed_spots.append(processed_spot)

#     # Update the displayed spots in the session
#     session['displayed_spots'] = displayed_spots + processed_spots

#     return render_template('recommendations.html', spots=processed_spots)



# # logging.basicConfig(level=logging.INFO)


# # def scrape_details(place_name):
# #     try:
# #         summary = wikipedia.summary(place_name)
# #     except wikipedia.exceptions.DisambiguationError as e:
# #         try:
# #             summary = wikipedia.summary(e.options[0])
# #         except Exception:
# #             return None
# #     except wikipedia.exceptions.PageError:
# #         return None
# #     except Exception:
# #         return None

# #     if summary:
# #         return summarize_text(summary)
# #     return None



# # def get_sentence_embeddings(sentences):
# #     # Load pre-trained BERT model and tokenizer
# #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# #     model = BertModel.from_pretrained('bert-base-uncased')
    
# #     # Tokenize sentences
# #     inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
# #     with torch.no_grad():
# #         outputs = model(**inputs)
    
# #     # Get the embeddings for the [CLS] token
# #     embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
# #     return embeddings

# # def summarize_text(text, num_clusters=4):
# #     # Tokenize the text into sentences
# #     sentences = sent_tokenize(text)
    
# #     # Check if the number of sentences is less than or equal to the number of clusters
# #     if len(sentences) <= num_clusters:
# #         return text  # Return the original text if it's too short
    
# #     # Get embeddings for each sentence
# #     embeddings = get_sentence_embeddings(sentences)
    
# #     # Perform K-means clustering
# #     kmeans = KMeans(n_clusters=num_clusters)
# #     kmeans.fit(embeddings)
    
# #     # Get cluster labels for each sentence and cluster centers
# #     cluster_labels = kmeans.labels_
# #     cluster_centers = kmeans.cluster_centers_
    
# #     # Group sentences and their embeddings by their cluster
# #     clustered_sentences = {i: [] for i in range(num_clusters)}
# #     clustered_embeddings = {i: [] for i in range(num_clusters)}
# #     for sentence, embedding, label in zip(sentences, embeddings, cluster_labels):
# #         clustered_sentences[label].append(sentence)
# #         clustered_embeddings[label].append(embedding)
    
# #     # Select the sentence closest to the centroid from each cluster
# #     summary_sentences = []
# #     for i in range(num_clusters):
# #         if clustered_sentences[i]:  # Check if the cluster is not empty
# #             distances = [np.linalg.norm(emb - cluster_centers[i]) for emb in clustered_embeddings[i]]
# #             closest_idx = np.argmin(distances)
# #             summary_sentences.append(clustered_sentences[i][closest_idx])
    
# #     # Join the summary sentences into a final summary
# #     summary = ' '.join(summary_sentences)
    
# #     return summary



# def scrape_details(place_name):
#     try:
#         summary = wikipedia.summary(place_name)
#     except wikipedia.exceptions.DisambiguationError as e:
#         try:
#             summary = wikipedia.summary(e.options[0])
#         except Exception:
#             return None
#     except wikipedia.exceptions.PageError:
#         return None
#     except Exception:
#         return None

#     if summary:
#         return summarize_text(summary)
#     return None


# def get_sentence_embeddings(sentences):
#     # Load pre-trained BERT model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
#     # Tokenize sentences
#     inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Get the hidden states
#     hidden_states = outputs.hidden_states

#     # Get the second-to-last layer
#     second_to_last_layer = hidden_states[-2]

#     # Average the second-to-last layer, excluding [CLS] and [SEP] tokens
#     sentence_embeddings = []
#     for i, sent in enumerate(sentences):
#         tokens = tokenizer.tokenize(sent)
#         token_embeddings = second_to_last_layer[i, 1:len(tokens)+1]  # Exclude [CLS] and [SEP]
#         sentence_embedding = torch.mean(token_embeddings, dim=0)
#         sentence_embeddings.append(sentence_embedding)

#     # Stack all sentence embeddings
#     embeddings = torch.stack(sentence_embeddings).numpy()
    
#     return embeddings


# def summarize_text(text, num_clusters=4):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)
    
#     # Check if the number of sentences is less than or equal to the number of clusters
#     if len(sentences) <= num_clusters:
#         return text  # Return the original text if it's too short
    
#     try:
#         # Get embeddings for each sentence
#         embeddings = get_sentence_embeddings(sentences)
        
#         # Perform K-means clustering
#         kmeans = KMeans(n_clusters=num_clusters)
#         kmeans.fit(embeddings)
        
#         # Get cluster labels for each sentence and cluster centers
#         cluster_labels = kmeans.labels_
#         cluster_centers = kmeans.cluster_centers_
        
#         # Group sentences and their embeddings by their cluster
#         clustered_sentences = {i: [] for i in range(num_clusters)}
#         clustered_embeddings = {i: [] for i in range(num_clusters)}
#         for sentence, embedding, label in zip(sentences, embeddings, cluster_labels):
#             clustered_sentences[label].append(sentence)
#             clustered_embeddings[label].append(embedding)
        
#         # Select the sentence closest to the centroid from each cluster
#         summary_sentences = []
#         for i in range(num_clusters):
#             if clustered_sentences[i]:  # Check if the cluster is not empty
#                 distances = [np.linalg.norm(emb - cluster_centers[i]) for emb in clustered_embeddings[i]]
#                 closest_idx = np.argmin(distances)
#                 summary_sentences.append(clustered_sentences[i][closest_idx])
        
#         # Join the summary sentences into a final summary
#         summary = ' '.join(summary_sentences)
        
#         return summary
#     except Exception:
#         # If any error occurs during summarization, return the original text
#         return text








# @app.route('/find_spots', methods=['POST'])
# @login_required
# def find_spots():
#     latitude = float(request.form['latitude'])
#     longitude = float(request.form['longitude'])
#     language = request.form['language']
#     user_radius = request.form.get('user_radius')
#     custom_user_radius = request.form.get('custom_user_radius')

#     if user_radius == 'custom':
#         user_radius = int(custom_user_radius)
#     else:
#         user_radius = int(user_radius)

#     # Fetch places within a large radius (to ensure enough results)
#     places = get_place_details(latitude, longitude)

#     # Filter out places without essential data and those with insufficient ratings
#     filtered_places = [
#         place for place in places
#         if 'name' in place and 'geometry' in place and 'location' in place['geometry'] and place.get('user_ratings_total', 0) > 200
#     ]

#     # Calculate distance for each place from the user's location
#     places_with_distance = [
#         (place, haversine(longitude, latitude, place['geometry']['location']['lng'], place['geometry']['location']['lat']))
#         for place in filtered_places
#     ]

#     # Filter places based on the user-specified radius
#     within_radius_places = [place for place in places_with_distance if place[1] <= user_radius]

#     # Sort places by distance
#     within_radius_places.sort(key=lambda x: x[1])

#     # Select top spots within the radius
#     top_spots = within_radius_places[:10]

#     # Initialize translation client
#     translated_spots = []
#     client = texttospeech.TextToSpeechClient()
#     for spot, distance in top_spots:
#         place_id = spot['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         # Translate place details
#         name_translation = translate_client.translate(spot['name'], target_language=language)
#         vicinity_translation = translate_client.translate(spot.get('vicinity', ''), target_language=language)
        
#         # Scrape and summarize description, if available
#         description = scrape_details(spot['name'])
#         if not description:
#             description = details.get('editorial_summary', {}).get('overview', 'No detailed description available.')
#         else:
#             description = summarize_text(description)
#         description_translation = translate_client.translate(description, target_language=language)

#         # Truncate the translated description to fit the limit
#         truncated_description = truncate_text(description_translation['translatedText'], 5000)

#         # Handle photos
#         photo_reference = None
#         if 'photos' in spot:
#             photo_reference = spot['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         # Get rating and reviews
#         rating = details.get('rating', 'No rating')
#         reviews = details.get('reviews', [])

#         # Append translated spot to the list
#         translated_spots.append({
#             'place_id': spot['place_id'],
#             'name': name_translation['translatedText'],
#             'vicinity': vicinity_translation['translatedText'],
#             'description': truncated_description,
#             'distance': distance,
#             'photo_url': photo_url,
#             'rating': rating,
#             'reviews': reviews
#         })

#          # Store translated_spots in session
#         session['translated_spots'] = [spot['place_id'] for spot in translated_spots]

#         # Generate audio for the description
#         text = truncated_description
#         synthesis_input = texttospeech.SynthesisInput(text=text)
#         voice = texttospeech.VoiceSelectionParams(language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
#         audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#         tts_response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

#         # Save audio file
#         audio_dir = "C:/Users/DishaDiya/Downloads/final/code/static/audio/"
#         os.makedirs(audio_dir, exist_ok=True)
#         audio_filename = f"{audio_dir}{spot['name'].replace(' ', '_')}_description.mp3"

#         with open(audio_filename, "wb") as out:
#             out.write(tts_response.audio_content)

#         # Add audio file link to the spot data
#         translated_spots[-1]['audio'] = f"audio/{spot['name'].replace(' ', '_')}_description.mp3"

#     # Get nearby cafes and restaurants within the user-specified radius
#     location = {'lat': latitude, 'lng': longitude}
#     cafes = get_nearby_places(location, 'cafe',70000)
#     restaurants = get_nearby_places(location, 'restaurant', 70000)

#     # Get start location details
#     start_location_details = get_place_details(latitude, longitude, search_radius=50)
#     start_location = {
#         'name': start_location_details[0].get('name', 'Unknown Location') if start_location_details else 'Unknown Location',
#         'vicinity': start_location_details[0].get('vicinity', '') if start_location_details else '',
#         'photo_url': start_location_details[0].get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400') if start_location_details else 'https://via.placeholder.com/400',
#         'rating': start_location_details[0].get('rating', 'No rating') if start_location_details else 'No rating'
#     }

#     liked_place_ids = [place.place_id for place in LikedPlace.query.filter_by(user_id=current_user.id).all()]

#     return render_template('results.html', spots=translated_spots, cafes=cafes, restaurants=restaurants, start_location=start_location, liked_place_ids=liked_place_ids)


# def truncate_text(text, max_bytes):
#     """Truncate text to ensure it does not exceed the specified number of bytes."""
#     encoded_text = text.encode('utf-8')
#     if len(encoded_text) <= max_bytes:
#         return text

#     truncated_text = encoded_text[:max_bytes].decode('utf-8', errors='ignore')
#     return truncated_text



# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places

# def get_nearby_places(location, place_type, radius=1000):
#     url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
#     response = requests.get(url)
#     results = response.json().get('results', [])

#     places = []
#     for place in results:
#         place_id = place['place_id']
#         details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
#         details = details_response.json().get('result', {})

#         photo_reference = None
#         if 'photos' in place:
#             photo_reference = place['photos'][0]['photo_reference']
#             photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
#         else:
#             photo_url = "https://via.placeholder.com/400"

#         places.append({
#             'name': details.get('name', 'Unknown'),
#             'vicinity': details.get('vicinity', 'Unknown'),
#             'rating': details.get('rating', 'No rating'),
#             'photo_url': photo_url,
#             'reviews': details.get('reviews', [])
#         })

#     return places


# @app.route('/about')
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, host='127.0.0.1', port=5003)














import datetime
import os
from sqlite3 import IntegrityError
from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from math import radians, cos, sin, asin, sqrt
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.cloud import translate_v2 as translate
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import wikipedia
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
import nltk
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.tokenize import sent_tokenize
from flask import jsonify
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')
from datetime import datetime, timezone
from sklearn.cluster import KMeans



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

GOOGLE_PLACES_API_KEY = 'AIzaSyC6JXdrY5SNL31rPWL1RUrln15ymEolLWQ'
GOOGLE_APPLICATION_CREDENTIALS = 'C:/Users/DishaDiya/Downloads/final_Edited/final/code/tourismapp-428107-fe80d138c6d4.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

translate_client = translate.Client()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    preferences = db.Column(db.String(500), nullable=True, default='')
    activities = db.Column(db.String(500), nullable=True, default='')
    ratings = db.Column(db.String(500), nullable=True, default='')

class LikedPlace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    place_id = db.Column(db.String(100), nullable=False)
    place_name = db.Column(db.String(200), nullable=False)
    place_vicinity = db.Column(db.String(300))
    place_rating = db.Column(db.Float)
    place_photo_url = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))

    user = db.relationship('User', backref=db.backref('liked_places', lazy=True))
    __table_args__ = (db.UniqueConstraint('user_id', 'place_id', name='_user_place_uc'),)

@app.route('/like_place', methods=['POST'])
@login_required
def like_place():
    place_id = request.form.get('place_id')
    place_name = request.form.get('place_name')
    place_vicinity = request.form.get('place_vicinity')
    place_rating = request.form.get('place_rating')
    place_photo_url = request.form.get('place_photo_url')

    existing_like = LikedPlace.query.filter_by(user_id=current_user.id, place_id=place_id).first()

    if existing_like:
        db.session.delete(existing_like)
        db.session.commit()
        if 'liked_places' in session:
            session['liked_places'] = [place for place in session['liked_places'] if place['place_id'] != place_id]
            session.modified = True
        return jsonify({'status': 'unliked'})
    else:
        new_like = LikedPlace(
            user_id=current_user.id,
            place_id=place_id,
            place_name=place_name,
            place_vicinity=place_vicinity,
            place_rating=place_rating,
            place_photo_url=place_photo_url,
            timestamp=datetime.now(timezone.utc)
        )
        try:
            db.session.add(new_like)
            db.session.commit()
            
            if 'liked_places' not in session:
                session['liked_places'] = []
            
            session['liked_places'].append({
                'place_id': place_id,
                'place_name': place_name,
                'place_vicinity': place_vicinity,
                'place_rating': place_rating,
                'place_photo_url': place_photo_url
            })
            session.modified = True
            
            return jsonify({'status': 'liked'})
        except IntegrityError:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': 'This place is already liked'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/liked_places')
@login_required
def liked_places():
    liked_places = LikedPlace.query.filter_by(user_id=current_user.id).order_by(LikedPlace.timestamp.desc()).all()
    return render_template('liked_places.html', liked_places=liked_places)


class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    place_id = db.Column(db.String(100), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'rate'
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    rating = db.Column(db.Float, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def get_place_details(lat, lng, search_radius=70000):
    params = {
        'location': f'{lat},{lng}',
        'radius': search_radius,  # Search within the specified radius
        'key': GOOGLE_PLACES_API_KEY,
        'type': 'tourist_attraction'
    }
    response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json', params=params)
    return response.json().get('results', [])



@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        preferences = request.form.get('preferences') 
        activities = request.form.get('activities')  # Get activities
        ratings = request.form.get('ratings')  # Corrected field name here

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(username=username, email=email, password=hashed_password,
                        preferences=preferences, activities=activities, ratings=ratings)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    if request.method == 'POST':
        if 'username' in request.form:
            # Updating personal information
            current_user.username = request.form['username']
            current_user.email = request.form['email']
            if request.form['new_password']:
                current_user.set_password(request.form['new_password'])
        
        # Always include username and email when updating
        username = current_user.username
        email = current_user.email
        
        if 'preferences' in request.form:
            current_user.preferences = request.form['preferences']
        
        if 'activities' in request.form:
            current_user.activities = request.form['activities']
        
        if 'accommodation_rating' in request.form:
            current_user.ratings = request.form['accommodation_rating']
        
        # Update the user in the database
        db.session.commit()
        
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    
    return render_template('account.html', title='Account')



def preprocess_user_data(user):
    preference_mapping = {
        'amusement park': ['amusement_park', 'theme_park', 'water_park'],
        'beach': ['beach', 'coast', 'seaside', 'ocean'],
        'mountain': ['mountain', 'hill', 'peak', 'alpine'],
        'city': ['city', 'urban', 'metropolis'],
        'historical': ['historical', 'ancient', 'heritage', 'museum'],
        'nature': ['nature', 'park', 'forest', 'wildlife'],
        # Add more mappings as needed
    }
    
    raw_preferences = [pref.strip().lower() for pref in user.preferences.split(',')] if user.preferences else []
    preferences = []
    for pref in raw_preferences:
        preferences.extend(preference_mapping.get(pref, [pref]))
    
    activities = [act.strip().lower() for act in user.activities.split(',')] if user.activities else []
    ratings = [int(r) for r in user.ratings.split(',')] if user.ratings else []
    return preferences, activities, ratings
    return preferences, activities, ratings

MIN_REVIEWS_THRESHOLD= 100

def get_recommendations(user, spots, start_location, num_recommendations=10, displayed_spots=None):
    if displayed_spots is None:
        displayed_spots = set()
    else:
        displayed_spots = set(spot['place_id'] for spot in displayed_spots)

    user_preferences, user_activities, user_ratings = preprocess_user_data(user)
    
    translated_spots = set(session.get('translated_spots', []))
    # Filter spots based on the minimum number of reviews, 70km radius, and not already displayed
    famous_spots = [
        spot for spot in spots 
        if spot.get('user_ratings_total', 0) >= MIN_REVIEWS_THRESHOLD
        and haversine(
            start_location['lat'], start_location['lng'],
            spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
        ) <= 70
        and spot['place_id'] not in displayed_spots
    ]
    
    if not famous_spots:
        return []  # Return empty list if no famous spots are found

    return content_based_filtering(user_preferences, user_activities, user_ratings, famous_spots, start_location, num_recommendations)




def content_based_filtering(user_preferences, user_activities, user_ratings, famous_spots, start_location, num_recommendations):
    place_scores = []
    for spot in famous_spots:
        score = 0
        types = spot.get('types', [])
        name = spot.get('name', '').lower()
        vicinity = spot.get('vicinity', '').lower()

        # Check for preference matches
        preference_match = any(pref in types or pref in name or pref in vicinity for pref in user_preferences)
        
        # Consider all spots, but give a significant boost to those matching preferences
        if preference_match:
            score += 50  # Significant boost for matching preferences

        # Check activities
        for activity in user_activities:
            if activity in types or activity in name or activity in vicinity:
                score += 5  # Moderate score for matching activities

        # Check rating
        spot_rating = spot.get('rating', 0)
        if spot_rating:
            user_rating_preference = max(user_ratings) if user_ratings else 5
            score += (min(5, round(spot_rating)) / user_rating_preference) * 3

        # Distance score (closer places get higher scores)
        distance = haversine(
            start_location['lat'], start_location['lng'],
            spot['geometry']['location']['lat'], spot['geometry']['location']['lng']
        )
        distance_score = 1 - (distance / 70)  # Normalize distance score
        score += distance_score * 10  # Give significant weight to distance

        place_scores.append((spot, score))

    # Rank and return top recommendations
    ranked_spots = sorted(place_scores, key=lambda x: x[1], reverse=True)
    return ranked_spots[:num_recommendations]

    # Rank and return top recommendations
    ranked_spots = sorted(place_scores, key=lambda x: x[1], reverse=True)
    return ranked_spots[:num_recommendations]



@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    start_location = {'lat': latitude, 'lng': longitude}

    # Get the displayed spots from the session
    displayed_spots = session.get('displayed_spots', [])

    # Get the translated spots from the session
    translated_spots = set(session.get('translated_spots', []))

    all_spots = get_place_details(latitude, longitude, search_radius=70000)  # 70km in meters
    
    # Log user preferences and all spots for debugging
    app.logger.info(f"User preferences: {current_user.preferences}")
    app.logger.info(f"All spots: {[spot['name'] for spot in all_spots]}")

    recommended_spots = get_recommendations(current_user, all_spots, start_location, displayed_spots=displayed_spots)

    # Log recommended spots for debugging
    app.logger.info(f"Recommended spots: {[spot[0]['name'] for spot in recommended_spots]}")

    processed_spots = []
    for spot, score in recommended_spots:
        if spot['place_id'] not in translated_spots:
            processed_spot = {
                'place_id': spot['place_id'],
                'name': spot['name'],
                'vicinity': spot.get('vicinity', 'No address available'),
                'rating': spot.get('rating', 'No rating'),
                'photo_url': spot.get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400'),
                'distance': haversine(latitude, longitude, spot['geometry']['location']['lat'], spot['geometry']['location']['lng'])
            }
            if processed_spot['photo_url'] != 'https://via.placeholder.com/400':
                processed_spot['photo_url'] = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={processed_spot['photo_url']}&key={GOOGLE_PLACES_API_KEY}"
            processed_spots.append(processed_spot)

    # Update the displayed spots in the session
    session['displayed_spots'] = displayed_spots + processed_spots

    return render_template('recommendations.html', spots=processed_spots)



def scrape_details(place_name):
    try:
        summary = wikipedia.summary(place_name)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0])
        except Exception:
            return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception:
        return None

    if summary:
        return summarize_text(summary)
    return None


def get_sentence_embeddings(sentences):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
    # Tokenize sentences
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the hidden states
    hidden_states = outputs.hidden_states

    # Get the second-to-last layer
    second_to_last_layer = hidden_states[-2]

    # Average the second-to-last layer, excluding [CLS] and [SEP] tokens
    sentence_embeddings = []
    for i, sent in enumerate(sentences):
        tokens = tokenizer.tokenize(sent)
        token_embeddings = second_to_last_layer[i, 1:len(tokens)+1]  # Exclude [CLS] and [SEP]
        sentence_embedding = torch.mean(token_embeddings, dim=0)
        sentence_embeddings.append(sentence_embedding)

    # Stack all sentence embeddings
    embeddings = torch.stack(sentence_embeddings).numpy()
    
    return embeddings


def summarize_text(text, num_clusters=4):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Check if the number of sentences is less than or equal to the number of clusters
    if len(sentences) <= num_clusters:
        return text  # Return the original text if it's too short
    
    try:
        # Get embeddings for each sentence
        embeddings = get_sentence_embeddings(sentences)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(embeddings)
        
        # Get cluster labels for each sentence and cluster centers
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        # Group sentences and their embeddings by their cluster
        clustered_sentences = {i: [] for i in range(num_clusters)}
        clustered_embeddings = {i: [] for i in range(num_clusters)}
        for sentence, embedding, label in zip(sentences, embeddings, cluster_labels):
            clustered_sentences[label].append(sentence)
            clustered_embeddings[label].append(embedding)
        
        # Select the sentence closest to the centroid from each cluster
        summary_sentences = []
        for i in range(num_clusters):
            if clustered_sentences[i]:  # Check if the cluster is not empty
                distances = [np.linalg.norm(emb - cluster_centers[i]) for emb in clustered_embeddings[i]]
                closest_idx = np.argmin(distances)
                summary_sentences.append(clustered_sentences[i][closest_idx])
        
        # Join the summary sentences into a final summary
        summary = ' '.join(summary_sentences)
        
        return summary
    except Exception:
        # If any error occurs during summarization, return the original text
        return text


@app.route('/find_spots', methods=['POST'])
@login_required
def find_spots():
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    language = request.form['language']
    user_radius = request.form.get('user_radius')
    custom_user_radius = request.form.get('custom_user_radius')

    if user_radius == 'custom':
        user_radius = int(custom_user_radius)
    else:
        user_radius = int(user_radius)

    # Fetch places within a large radius (to ensure enough results)
    places = get_place_details(latitude, longitude)

    # Filter out places without essential data and those with insufficient ratings
    filtered_places = [
        place for place in places
        if 'name' in place and 'geometry' in place and 'location' in place['geometry'] and place.get('user_ratings_total', 0) > 200
    ]

    # Calculate distance for each place from the user's location
    places_with_distance = [
        (place, haversine(longitude, latitude, place['geometry']['location']['lng'], place['geometry']['location']['lat']))
        for place in filtered_places
    ]

    # Filter places based on the user-specified radius
    within_radius_places = [place for place in places_with_distance if place[1] <= user_radius]

    # Sort places by distance
    within_radius_places.sort(key=lambda x: x[1])

    # Select top spots within the radius
    top_spots = within_radius_places[:10]

    # Initialize translation client
    translated_spots = []
    client = texttospeech.TextToSpeechClient()
    for spot, distance in top_spots:
        place_id = spot['place_id']
        details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
        details = details_response.json().get('result', {})

        # Translate place details
        name_translation = translate_client.translate(spot['name'], target_language=language)
        vicinity_translation = translate_client.translate(spot.get('vicinity', ''), target_language=language)
        
        # Scrape and summarize description, if available
        description = scrape_details(spot['name'])
        if not description:
            description = details.get('editorial_summary', {}).get('overview', 'No detailed description available.')
        else:
            description = summarize_text(description)
        description_translation = translate_client.translate(description, target_language=language)

        # Truncate the translated description to fit the limit
        truncated_description = truncate_text(description_translation['translatedText'], 5000)

        # Handle photos
        photo_reference = None
        if 'photos' in spot:
            photo_reference = spot['photos'][0]['photo_reference']
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
        else:
            photo_url = "https://via.placeholder.com/400"

        # Get rating and reviews
        rating = details.get('rating', 'No rating')
        reviews = details.get('reviews', [])

        # Append translated spot to the list
        translated_spots.append({
            'name': name_translation['translatedText'],
            'vicinity': vicinity_translation['translatedText'],
            'description': truncated_description,
            'distance': distance,
            'photo_url': photo_url,
            'rating': rating,
            'reviews': reviews,
            'place_id': spot['place_id']
        })


        session['translated_spots'] = [spot['place_id'] for spot in translated_spots]

        # Generate audio for the description
        # Generate audio for the description
        text = truncated_description
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
        language_code=language,  # Use the specified language
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        tts_response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        # Save audio file
        audio_dir = "C:/Users/DishaDiya/Downloads/final/code/static/audio/"
        os.makedirs(audio_dir, exist_ok=True)
        audio_filename = f"{audio_dir}{spot['name'].replace(' ', '_')}_description.mp3"

        with open(audio_filename, "wb") as out:
            out.write(tts_response.audio_content)

        # Add audio file link to the spot data
        translated_spots[-1]['audio'] = f"audio/{spot['name'].replace(' ', '_')}_description.mp3"

    # Get nearby cafes and restaurants within the user-specified radius
    location = {'lat': latitude, 'lng': longitude}
    cafes = get_nearby_places(location, 'cafe',70000)
    restaurants = get_nearby_places(location, 'restaurant', 70000)

    # Get start location details
    start_location_details = get_place_details(latitude, longitude, search_radius=50)
    start_location = {
        'name': start_location_details[0].get('name', 'Unknown Location') if start_location_details else 'Unknown Location',
        'vicinity': start_location_details[0].get('vicinity', '') if start_location_details else '',
        'photo_url': start_location_details[0].get('photos', [{'photo_reference': None}])[0].get('photo_reference', 'https://via.placeholder.com/400') if start_location_details else 'https://via.placeholder.com/400',
        'rating': start_location_details[0].get('rating', 'No rating') if start_location_details else 'No rating'
    }

    liked_place_ids = [place.place_id for place in LikedPlace.query.filter_by(user_id=current_user.id).all()]

    return render_template('results.html', spots=translated_spots, cafes=cafes, restaurants=restaurants, start_location=start_location)


def truncate_text(text, max_bytes):
    """Truncate text to ensure it does not exceed the specified number of bytes."""
    encoded_text = text.encode('utf-8')
    if len(encoded_text) <= max_bytes:
        return text

    truncated_text = encoded_text[:max_bytes].decode('utf-8', errors='ignore')
    return truncated_text



def get_nearby_places(location, place_type, radius=1000):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
    response = requests.get(url)
    results = response.json().get('results', [])

    places = []
    for place in results:
        place_id = place['place_id']
        details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
        details = details_response.json().get('result', {})

        photo_reference = None
        if 'photos' in place:
            photo_reference = place['photos'][0]['photo_reference']
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
        else:
            photo_url = "https://via.placeholder.com/400"

        places.append({
            'name': details.get('name', 'Unknown'),
            'vicinity': details.get('vicinity', 'Unknown'),
            'rating': details.get('rating', 'No rating'),
            'photo_url': photo_url,
            'reviews': details.get('reviews', [])
        })

    return places

def get_nearby_places(location, place_type, radius=1000):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location['lat']},{location['lng']}&radius={radius}&type={place_type}&key={GOOGLE_PLACES_API_KEY}"
    response = requests.get(url)
    results = response.json().get('results', [])

    places = []
    for place in results:
        place_id = place['place_id']
        details_response = requests.get(f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}')
        details = details_response.json().get('result', {})

        photo_reference = None
        if 'photos' in place:
            photo_reference = place['photos'][0]['photo_reference']
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
        else:
            photo_url = "https://via.placeholder.com/400"

        places.append({
            'name': details.get('name', 'Unknown'),
            'vicinity': details.get('vicinity', 'Unknown'),
            'rating': details.get('rating', 'No rating'),
            'photo_url': photo_url,
            'reviews': details.get('reviews', [])
        })

    return places


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='127.0.0.1', port=5003)


