import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import os
import logging
from typing import Dict, Callable, Optional
from config import config

logger = logging.getLogger(__name__)
filepath = "/workspaces/Customer-Support-Automation/data/twcs.csv"

class LargeFileProcessor:
    """Handles processing of large CSV files (500MB+) efficiently"""
    
    def __init__(self, chunk_size: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.client = None
        self.processed_chunks = 0
        self.total_rows = 0
        
    def initialize_dask_client(self):
        """Initialize Dask client for distributed processing"""
        try:
            self.client = Client(n_workers=config.MAX_WORKERS, threads_per_worker=2)
            logger.info(f"Dask client initialized with {config.MAX_WORKERS} workers")
        except Exception as e:
            logger.warning(f"Failed to initialize Dask client: {e}")
            self.client = None
    
    def estimate_file_size(self, filepath: str) -> Dict:
        """Estimate file characteristics"""
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        
        # Read first few lines to estimate structure
        sample_df = pd.read_csv(filepath, nrows=1000)
        
        estimated_rows = int(file_size * 1000 / sample_df.memory_usage(deep=True).sum() * 1024 * 1024)
        
        return {
            'file_size_mb': file_size,
            'estimated_rows': estimated_rows,
            'columns': list(sample_df.columns),
            'recommended_chunk_size': min(self.chunk_size, max(1000, estimated_rows // 100))
        }
    
    def process_large_csv(self, filepath: str, callback_func: Optional[Callable] = None) -> pd.DataFrame:
        """Process large CSV file in chunks"""
        
        logger.info(f"Starting to process large file: {filepath}")
        file_info = self.estimate_file_size(filepath)
        logger.info(f"File info: {file_info}")
        
        # Use Dask for very large files
        if file_info['file_size_mb'] > 100:
            return self._process_with_dask(filepath)
        else:
            return self._process_with_chunks(filepath, callback_func)
    
    def _process_with_dask(self, filepath: str) -> pd.DataFrame:
        """Process using Dask for distributed computing"""
        try:
            if not self.client:
                self.initialize_dask_client()
            
            # Read with Dask
            df = dd.read_csv(filepath)
            
            # Basic preprocessing
            df['created_at'] = dd.to_datetime(df['created_at'])
            df['clean_text'] = df['text'].str.replace(r'@\w+', '', regex=True).str.strip()
            
            # Compute and return
            result = df.compute()
            logger.info(f"Processed {len(result)} rows using Dask")
            return result
            
        except Exception as e:
            logger.error(f"Dask processing failed: {e}")
            return self._process_with_chunks(filepath)
    
    def _process_with_chunks(self, filepath: str, callback_func: Optional[Callable] = None) -> pd.DataFrame:
        """Process file in chunks"""
        chunks = []
        
        try:
            chunk_reader = pd.read_csv(filepath, chunksize=self.chunk_size)
            
            for i, chunk in enumerate(chunk_reader):
                # Process chunk
                chunk['created_at'] = pd.to_datetime(chunk['created_at'])
                chunk['clean_text'] = chunk['text'].str.replace(r'@\w+', '', regex=True).str.strip()
                
                chunks.append(chunk)
                self.processed_chunks += 1
                self.total_rows += len(chunk)
                
                if callback_func:
                    callback_func(i, len(chunk), self.total_rows)
                
                if i % 10 == 0:
                    logger.info(f"Processed {self.processed_chunks} chunks, {self.total_rows} total rows")
            
            # Combine all chunks
            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"Successfully processed {len(result)} rows from large file")
            return result
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            raise