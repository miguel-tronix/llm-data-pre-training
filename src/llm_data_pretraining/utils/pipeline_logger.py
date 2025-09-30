import logging
import os

class PipelineLogger:
    _logger = None
    
    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger("pretraining_pipeline")
            cls._logger.setLevel(logging.DEBUG)
            
            if not cls._logger.handlers:
                file_handler = logging.FileHandler(os.getenv("LOG_FILE", "logs/pretraining_pipeline.log"))
                file_handler.setLevel(os.getenv("LOG_LEVEL", "INFO"))
                
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                cls._logger.addHandler(file_handler)
        
        return cls._logger

# Convenience function
def get_pipeline_logger():
    return PipelineLogger.get_logger()