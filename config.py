import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for the entire system"""
    
    def __init__(self):
       # self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///support.db')
       # self.REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '10000'))
        self.MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
        self.MODEL_PATH = os.getenv('MODEL_PATH', './models/')
        self.DATA_PATH = os.getenv('DATA_PATH', './data/')
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Create directories
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)

config = Config()