import sys
from joblib import Memory

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

print(n_jobs)
