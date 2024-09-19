import os
import json

def save_results(query, metric, results, filename, limit):
    with open(filename, 'w') as f:
        json.dump({"query": query, "metric": metric, "limit": limit, "results": results}, f, indent=2)

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)
