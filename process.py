import multiprocessing

def worker(num):
  print("Worker process", num)

processes = []
for i in range(5):
  process = multiprocessing.Process(target=worker, args=(i,))
  processes.append(process)
  process.start()

for process in processes:
  process.join()