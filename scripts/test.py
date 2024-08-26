import sys
from joblib import Memory
import multiprocessing

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

print('n_jobs: ', n_jobs)
print('nombre cpus :', multiprocessing.cpu_count())
