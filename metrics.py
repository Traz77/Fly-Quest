from prometheus_client import Counter, Gauge, start_http_server

episodes_total = Counter('flyq_episodes_total', 'Total training episodes completed')
wins_total = Counter('flyq_wins_total', 'Total times Fly caught the Human')
spider_deaths_total = Counter('flyq_spider_deaths_total', 'Total times Fly hit a Spider')
timeouts_total = Counter('flyq_timeouts_total', 'Total episodes that timed out')

episode_reward = Gauge('flyq_episode_reward', 'Total reward for the latest episode')
episode_length = Gauge('flyq_episode_length', 'Steps taken in the latest episode')

def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Metrics server started on http://localhost:{port}")