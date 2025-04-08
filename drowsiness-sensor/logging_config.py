import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_performance(metrics):
    """Logs performance metrics to the log file."""
    logging.info(f"Performance - CPU: {metrics['cpu']:.1f}% | Memory: {metrics['memory']:.1f}MB | FPS: {metrics['fps']:.1f}")

def log_event(message, level="info"):
    """Logs general events with a specified log level."""
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "debug":
        logging.debug(message)
    elif level == "warning":
        logging.warning(message)