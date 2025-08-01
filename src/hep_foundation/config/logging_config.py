import logging

# Add custom PROGRESS log level (between INFO=20 and WARNING=30)
PROGRESS_LEVEL = 25
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")

# Add custom TEMPLOG log level (between WARNING=30 and ERROR=40)
TEMPLOG_LEVEL = 35
logging.addLevelName(TEMPLOG_LEVEL, "TEMPLOG")


def progress(self, message, *args, **kwargs):
    """Log a progress message at PROGRESS level."""
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)


def templog(self, message, *args, **kwargs):
    """Log a templog message at TEMPLOG level."""
    if self.isEnabledFor(TEMPLOG_LEVEL):
        self._log(TEMPLOG_LEVEL, message, args, **kwargs)


# Add the progress method to Logger class
logging.Logger.progress = progress
logging.Logger.templog = templog


def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration for the entire package

    Note: TensorFlow C++ logging level is controlled by TF_CPP_MIN_LOG_LEVEL
    environment variable which is set in hep_foundation.__init__.py before
    any TensorFlow imports can occur.
    """

    # Suppress specific TensorFlow Python warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

    # Set TensorFlow logging level
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)  # Only show ERROR messages
    except ImportError:
        # TensorFlow not available, skip TF logging configuration
        pass

    # Create formatter
    if level == logging.DEBUG:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    return root_logger


def get_logger(name):
    """Get a logger for a module"""
    return logging.getLogger(name)
