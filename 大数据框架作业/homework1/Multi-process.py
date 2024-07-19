import multiprocessing
from collections import defaultdict

def map_function(data_chunk):
    """Map function to process each chunk of data."""
    result = []
    for key, value in data_chunk:
        result.append((key, (value, 1)))
    return result

def reduce_function(intermediate_data):
    """Reduce function to compute the average from summed values."""
    totals = defaultdict(list)
    for key, value in intermediate_data:
        totals[key].append(value)
    
    final_result = {}
    for key, values in totals.items():
        total_sum = sum(val[0] for val in values)
        total_count = sum(val[1] for val in values)
        final_result[key] = total_sum / total_count
    return final_result

def distribute_computation(data, map_func, reduce_func, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    num_workers = min(num_workers, len(data))
    chunk_size = max(1, len(data) // num_workers)  
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Split the data into chunks
        data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Map phase: process each chunk in parallel
        map_results = pool.map(map_func, data_chunks)
        
        # Combine all intermediate results for the reduce phase
        combined_results = [item for sublist in map_results for item in sublist]
    
    # Reduce phase: compute final results
    final_result = reduce_func(combined_results)
    return final_result


def read_data_from_file(filename="data.txt"):
    """Read key-value pairs from a file and return as a list of tuples."""
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.strip().split(',')
            data.append((key, int(value)))
    return data
def save_results_to_file(results, filename="result.txt"):
    """Save the results dictionary to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        for key, value in results.items():
            file.write(f"{key},{value}\n")

if __name__ == "__main__":
    data = read_data_from_file()
    result = distribute_computation(data, map_function, reduce_function)
    print(result)
    save_results_to_file(result)
    
