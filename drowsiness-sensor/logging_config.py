import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_performance(metrics):
    """Logs performance metrics such as CPU and memory usage."""
    logging.info(f"CPU: {metrics['cpu']}% | Memory: {metrics['memory']}MB | FPS: {metrics['fps']}")