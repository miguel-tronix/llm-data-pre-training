import logging
import os


class PipelineLogger:
    _logger = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("pretraining_pipeline")
            cls._logger.setLevel(logging.DEBUG)

            if not cls._logger.handlers:
                log_file = os.getenv("LOG_FILE", "logs/pretraining_pipeline.log")
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(os.getenv("LOG_LEVEL", "INFO"))

                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)

                cls._logger.addHandler(file_handler)

        return cls._logger


# Convenience function
def get_pipeline_logger() -> logging.Logger:
    return PipelineLogger.get_logger()
