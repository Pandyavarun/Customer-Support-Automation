import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging
from typing import Dict
from config import config

logger = logging.getLogger(__name__)

class ModelBuilder:
    """Builds and manages ML models for the support system"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
    
    def build_response_classifier(self, text_features, y):
        """Build response category classifier"""
        X_train, X_test, y_train, y_test = train_test_split(
            text_features, y, test_size=0.2, random_state=42
        )
        
        # Try multiple algorithms
        models = {
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            logger.info(f"{name} accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.models['response_classifier'] = best_model
        logger.info(f"Best response classifier accuracy: {best_score:.3f}")
    
    def build_urgency_predictor(self, X_combined, y):
        """Build urgency level predictor"""
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        self.models['urgency_predictor'] = model
        logger.info(f"Urgency predictor accuracy: {score:.3f}")
    
    def build_sentiment_analyzer(self, text_features, y):
        """Build sentiment analyzer"""
        X_train, X_test, y_train, y_test = train_test_split(
            text_features, y, test_size=0.2, random_state=42
        )
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        self.models['sentiment_analyzer'] = model
        logger.info(f"Sentiment analyzer accuracy: {score:.3f}")
    
    def save_models(self):
        """Save all models and components"""
        logger.info("Saving models and components...")
        
        # Save models
        for name, model in self.models.items():
            filename = os.path.join(config.MODEL_PATH, f"{name}.joblib")
            joblib.dump(model, filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            filename = os.path.join(config.MODEL_PATH, f"vectorizer_{name}.joblib")
            joblib.dump(vectorizer, filename)
        
        logger.info("All models saved successfully")
    
    def load_models(self):
        """Load pre-trained models"""
        logger.info("Loading pre-trained models...")
        
        # Load models
        model_files = [f for f in os.listdir(config.MODEL_PATH) 
                      if f.endswith('.joblib') and not f.startswith('vectorizer_')]
        
        for model_file in model_files:
            name = model_file.replace('.joblib', '')
            filepath = os.path.join(config.MODEL_PATH, model_file)
            self.models[name] = joblib.load(filepath)
            logger.info(f"Loaded {name}")
        
        # Load vectorizers
        vectorizer_files = [f for f in os.listdir(config.MODEL_PATH) 
                          if f.startswith('vectorizer_')]
        
        for vec_file in vectorizer_files:
            name = vec_file.replace('vectorizer_', '').replace('.joblib', '')
            filepath = os.path.join(config.MODEL_PATH, vec_file)
            self.vectorizers[name] = joblib.load(filepath)
        
        logger.info("Models loaded successfully")