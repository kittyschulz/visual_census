import sys
from queue import Queue
from threading import Thread


class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""

    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""

    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


def main():
    """Download and extract all images from the UCF StreetView Dataset."""
    import urllib.request
    from zipfile import ZipFile

    def download_images(part_no):
        url = 'http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/zipped%20images/part{}.zip'.format(
            part_no)
        path = '../ucf_data/part{}.zip'.format(part_no)
        urllib.request.urlretrieve(url, path)
        print('downloading part {}'.format(part_no))

        with ZipFile(path, 'r') as zipObj:
            zipObj.extractall()
            print('unzipping part {}'.format(part_no))

    parts = list(range(1, 11))

    pool = ThreadPool(10)
    pool.map(download_images, parts)
    pool.wait_completion()


if __name__ == "__main__":
    main()
