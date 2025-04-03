## CPU-Bound vs. I/O-Bound Tasks 

> A task is CPU-bound if it is limited by the processing power of the CPU. 
> A task is I/O Bound when it spends most of its time waiting for external resources, 
such as disk operations, network requests, or database queries. 

Quick rule of thumb: if a task is CPU-bound, use **multiprocessing**. If it's I/O bound, use **multithreading / async programming**. 

CPUs handle sequential tasks efficiently. GPUs handle parallel computations efficiently. 