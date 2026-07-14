import multiprocessing as mp
from multiprocessing import cpu_count

PROCESSES = cpu_count() - 1

# The workers run a closure (process_task) that captures `job`. Closures can't be
# pickled, so this relies on the "fork" start method (child inherits parent memory)
# rather than "spawn" (the macOS/Windows default, which pickles the target).
_ctx = mp.get_context("fork")


class BatchJob:
    def __init__(self, num_processes=PROCESSES):
        self.num_processes = num_processes

    def __call__(self, job):
        def batched_job(arg_batch):
            arg_batch = list(arg_batch)

            in_queue = _ctx.JoinableQueue()
            out_queue = _ctx.Queue()

            def process_task(task_queue, return_queue):
                while True:
                    next_task = task_queue.get()
                    if next_task is None:
                        task_queue.task_done()
                        break
                    args_ind, args = next_task
                    result = job(args)
                    return_queue.put((args_ind, result))
                    task_queue.task_done()

            procs = [
                _ctx.Process(target=process_task, args=(in_queue, out_queue))
                for _ in range(self.num_processes)
            ]
            for p in procs:
                p.start()

            for ind, arg in enumerate(arg_batch):
                in_queue.put((ind, arg))
            for _ in range(self.num_processes):
                in_queue.put(None)

            results = [out_queue.get() for _ in range(len(arg_batch))]

            in_queue.join()
            for p in procs:
                p.join()

            results.sort(key=lambda item: item[0])
            return [r for _, r in results]

        return batched_job


if __name__ == "__main__":
    batch_job = BatchJob()

    @batch_job
    def job(arg):
        return arg * 2

    results = job([1, 2, 3])
    print(results)
