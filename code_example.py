import asyncio
import threading
from collections import deque
from typing import Any, Callable, Tuple
import signal
import sys


class HybridShutdownEvent:
    def __init__(self):
        self._threading_event=threading.Event()
        self._asyncio_event=asyncio.Event()
    def set(self):
        self._threading_event.set()
        try:
            loop=asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._asyncio_event.set)
        except RuntimeError:
            print(" from hybridShutdownEvent instance: no running asyncio event loop")
    def is_set(self):
        return self._threading_event.is_set() or self._asyncio_event.is_set()
    def threading(self):
        return self._threading_event
    def asyncio(self):
        return self._asyncio_event
        
    
class CrossLoopLockManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    async def acquire(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._lock_blocking_acquire)
        self._owner = threading.get_ident()

    def _lock_blocking_acquire(self):
        self._lock.acquire()

    async def release(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("Lock release from non-owner thread")
        self._owner = None
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._lock.release)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class CrossLoopQueue:
    def __init__(self, shutdown_event: HybridShutdownEvent):
        self._queue = deque()
        self._lock = CrossLoopLockManager()
        self._shutdown = shutdown_event

    async def put(self, item):
        async with self._lock:
            self._queue.append(item)

    async def get(self):
        while not self._shutdown.is_set():
            async with self._lock:
                if self._queue:
                    return self._queue.popleft()
            await asyncio.sleep(0.1)
        raise asyncio.CancelledError("Queue get cancelled by shutdown.")

    async def safe_len(self):
        async with self._lock:
            return len(self._queue)

    async def safe_empty(self):
        async with self._lock:
            return len(self._queue) == 0


class GracefulGenerator:
    """  through its method run the async generator will run unless there is
    a stop event on the way. The loop uses the special deque with the special
    lock so that many concurrent asyncio iterator may access this deque via this
    special lock. This is the job of do_work while  handle_item  is tasked with
    filling the results in the queue as they come."""
    def __init__(self, resource: CrossLoopQueue,shutdown_event=None):
        self.resource = resource
        self.shutdown = shutdown_event if shutdown_event is not None else asyncio.Event()

    def stop(self):
        self.shutdown.set()

    async def run(self, async_gen_func: Callable, *gen_args, **gen_kwargs):
        try:
            print(f"self.shutdown is set? {self.shutdown.is_set()}")
            print("ok we are to run do_work")
            await self.do_work(async_gen_func, gen_args, gen_kwargs)
            await asyncio.sleep(0)

            """
            while not self.shutdown.is_set():
                print("ok we are to run do_work")
                await self.do_work(async_gen_func, gen_args, gen_kwargs)
                await asyncio.sleep(0)
            """
        finally:
            await self.cleanup()

    async def do_work(
        self,
        async_gen_func: Callable,
        async_gen_args: Tuple = (),
        async_gen_kwargs: dict = {},
    ):
        #print("do_work started...")
        the_agen=async_gen_func(*async_gen_args, **async_gen_kwargs)
        try:
            
            async for result in the_agen:
                #print("we are in the loop of the async_gen_func")
                if self.shutdown.is_set():
                    #print("self.shutdown.asyncio is set actually")
                    break
                #print("we are to await handle_item")
                await self.handle_item(result)
        finally:
            await the_agen.aclose()
    async def handle_item(self, item: Any):
        #print(f"handle_item runs with {item}")
        await self.resource.put(item)

    async def cleanup(self):
        print("Cleaning up GracefulGenerator...")
        await asyncio.sleep(0)
        # Optionally handle resource-specific logic
""" this code starts a new loop to run an async iterator"""
def run_async_loop(lock_mgr,name):
    async def worker():
        for i in range(3):
            async with lock_mgr:
                print(f"{name} got lock at {i}")
                await asyncio.sleep(0.3)
                print(f"{name} releasing lock as {i}")
    loop=asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(worker())
""" this code  runs an async iterator in a dedicated thread that runs a dedicated
loop. The reason why this is: this decoration of the async iterator, that  should be run
through the graceful generator, is here because then the threads are run independently
from the main loop. The extra loops are explicitly set below in order to allow
more refined processes in the future. asyncio.run would do the whole job in a 
more concise way but not differently: it would simply create this_loop and close
it in the backyard. Here we could pass the loop in the return statement eventually
rather than close it for good"""
def launch_iterator_in_thread(the_gen,the_gen_args,the_gen_kwargs):
    this_loop=asyncio.new_event_loop()
    future=this_loop.create_future()
    def thread_target():
        asyncio.set_event_loop(this_loop)
        async def inner_wrapper():
            try:
                result=await the_gen(*the_gen_args,**the_gen_kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        try:
            this_loop.call_soon_threadsafe(asyncio.ensure_future,inner_wrapper())
            this_loop.run_forever()
        finally:
            this_loop.close()
    thread=threading.Thread(target=thread_target)
    thread.start()
    return thread

async def async_gen(n, rank=None):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield f"data-{i} for rank {rank}"
    
def start_new_event_loop_in_thread(coroutine):
    def runner():
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try: 
            loop.run_until_complete(coroutine)
        except asyncio.CancelledError:
            print("main coroutine was cancelled")
        finally:
            loop.close()
            print("chief async loop closed")
    thread=threading.Thread(target=runner,daemon=True)
    thread.start()
    return thread

async def consume_until_drained(queue: CrossLoopQueue,shutdown_event:HybridShutdownEvent):
    while True:
        try:
            item = await queue.get()
        except asyncio.TimeoutError:
            print("Queue is empty, retrying...")
            #if shutdown_event.is_set() and await queue.safe_len()==0:
            if await queue.safe_len()==0:
                print("shutdown event occured and queue is empty now. Closing...")
                break
            continue  
        except asyncio.CancelledError:
            break
        else: # runs only when try was a success so the else refers to except actually
            print("Got:", item)

async def main(shutdown_event=HybridShutdownEvent,n_gens=3):
    
    queue = CrossLoopQueue(shutdown_event=shutdown_event)
    """ this is the point where we can pile up together many async gens
     they would share the same queue, the same shutdown event"""
    gens=[GracefulGenerator(queue,shutdown_event) for _ in range(n_gens)]
    tasks = [asyncio.create_task(gen.run(async_gen, 5,rank=rank)) 
             for rank,gen in enumerate(gens)]
    """ trick made to secure that async gens are running"""
    await asyncio.sleep(0.1)
    
    print("we got to the pre clearing the queue")
    queue_length=await queue.safe_len()
    print(" queue length is ",queue_length)
    queue_consumer_task=asyncio.create_task( consume_until_drained(queue,shutdown_event))
    
    try:
        await asyncio.wait(tasks,return_when=asyncio.ALL_COMPLETED)
    except KeyboardInterrupt:
        print("interrupt received")
    finally:
        await asyncio.sleep(0.2)
        shutdown_event.set()
        
    await asyncio.gather(*tasks,queue_consumer_task,
                             return_exceptions=True)
    
def signal_handler(sig,frame):
    print("shutdown siganl received")
    shutdown_event.set()
    
if __name__=="__main__":
    shutdown_event=HybridShutdownEvent()
    signal.signal(signal.SIGINT,signal_handler)
    """ we create a thread that is  dedicated to execute main in an asyncio loop. However, since 
    this loop cannot run concurrently to the loop in the python interpreter,
    we have to step through start_new_event_loop_in_thread that will
    generate the thread and the asyncio loop within this thread. There, main()
    will call in the main ingredients: a controlled collection.deque that itself will be
    managed and controlled through the CrossLoopLockManager that is called from within
    the CrossLoopQueue: this class calls a deque() from collections and controls
    every entry through a CrossLoopLockManager. the CrossLoopQueue can be shared
    via the graceful generator  that expects to run an async iterator. Thus main()
    can call one single " secial deque()" and run many tasks in parallel that would
    all feed a shared queue here, the "special lock()" securing that these 
    that it will run through its own method 'run'. """
    thread=start_new_event_loop_in_thread(main(shutdown_event))
    try:
        thread.join()
    except KeyboardInterrupt:
        shutdown_event.set()
        thread.join()






