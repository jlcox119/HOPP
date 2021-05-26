import concurrent.futures as cf
import time

def func(wait):
    time.sleep(wait)
    return f'Done... {wait}'

start = time.perf_counter()

with cf.ThreadPoolExecutor(max_workers=6) as executor:
    tasks = {executor.submit(func, wait): wait for wait in range(10, 0, -1)}

    for future in cf.as_completed(tasks):
        wait = tasks[future]
        result = future.result()

        print(wait, result)

end = time.perf_counter()
print(f'Elapsed time: {end-start:.2f} secs')