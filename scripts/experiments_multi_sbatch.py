import subprocess
import threading
from enum import Enum

from tqdm.auto import tqdm

from rmrl.__main__ import parse_args, get_single_run_args_list


def main():
    # parse arguments for regular job
    all_run_args = parse_args()

    # use `num_workers` as limit to number of jobs that can run
    sem = threading.BoundedSemaphore(value=all_run_args.num_workers)  # semaphore to limit number of running jobs
    all_run_args.num_workers = 1  # running sbatch jobs. do not multiprocess

    single_run_args_list = get_single_run_args_list(all_run_args)

    all_jobs_pbar = tqdm(total=len(single_run_args_list), desc='running experiments as sbatch jobs')

    def on_job_end():
        sem.release()  # release semaphore and allow new job to be run
        all_jobs_pbar.update()  # update pbar that one job is complete

    for i, run_args in enumerate(single_run_args_list, 1):
        # prepare args for job
        popen_args = f'scripts/run_python_job.sh rmrl_exp_{i} 2 0 -m rmrl'.split()
        for arg_name, arg_val in vars(run_args).items():
            if arg_val is None:
                continue  # omit arg and use default
            popen_args += [f'--{arg_name}'] + __prep_arg(arg_val)

        # run job
        sem.acquire()  # stop at semaphore if number of jobs is at maximum
        popen_and_call(on_job_end, popen_args)

    all_jobs_pbar.close()


def __prep_arg(arg_val):
    if isinstance(arg_val, bool):  # booleans are "store_true" actions
        return []
    elif isinstance(arg_val, list) or isinstance(arg_val, tuple):
        return list(map(lambda v: __prep_arg(v)[0], arg_val))
    elif callable(arg_val):
        if hasattr(arg_val, '__name__'):
            return __prep_arg(arg_val.__name__)
        else:
            return __prep_arg(arg_val.__class__.__name__)
    elif isinstance(arg_val, Enum):
        return __prep_arg(arg_val.value)
    else:
        return [str(arg_val)]


# # ==================================================================================================
# https://stackoverflow.com/questions/2581817/python-subprocess-callback-when-cmd-exits
# ==================================================================================================
def popen_and_call(on_exit, popen_args):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    on_exit when the subprocess completes.
    on_exit is a callable object, and popen_args is a list/tuple of args that
    would give to subprocess.Popen.
    """

    def run_in_thread(on_exit, popen_args):
        proc = subprocess.Popen(*popen_args)
        proc.wait()
        on_exit()
        return

    thread = threading.Thread(target=run_in_thread, args=(on_exit, popen_args))
    thread.start()
    # returns immediately after the thread starts
    return thread


# ==================================================================================================


if __name__ == '__main__':
    main()
