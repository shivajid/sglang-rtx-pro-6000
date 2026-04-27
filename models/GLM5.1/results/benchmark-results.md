```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2310.09   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6424478   
Request throughput (req/s):              0.66      
Input token throughput (tok/s):          339.80    
Output token throughput (tok/s):         2785.55   
Peak output token throughput (tok/s):    4092.00   
Peak concurrent requests:                517       
Total token throughput (tok/s):          3125.35   
Concurrency:                             408.72    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   614696.48 
Median E2E Latency (ms):                 615683.13 
P90 E2E Latency (ms):                    1076596.45
P99 E2E Latency (ms):                    1230529.45
---------------Time to First Token----------------
Mean TTFT (ms):                          5190.39   
Median TTFT (ms):                        344.23    
P99 TTFT (ms):                           26701.62  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          155.26    
Median TPOT (ms):                        148.98    
P99 TPOT (ms):                           164.18    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           145.87    
Median ITL (ms):                         139.63    
P95 ITL (ms):                            204.98    
P99 ITL (ms):                            297.69    
Max ITL (ms):                            387708.06 
==================================================
```

### NVFP4 Results

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     512       
Benchmark duration (s):                  1459.78   
Total input tokens:                      258921    
Total input text tokens:                 258921    
Total generated tokens:                  2054943   
Total generated tokens (retokenized):    2053760   
Request throughput (req/s):              0.35      
Input token throughput (tok/s):          177.37    
Output token throughput (tok/s):         1407.71   
Peak output token throughput (tok/s):    770.00    
Peak concurrent requests:                512       
Total token throughput (tok/s):          1585.08   
Concurrency:                             249.24    
Accept length:                           3.21      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   710627.73 
Median E2E Latency (ms):                 695163.08 
P90 E2E Latency (ms):                    1263758.99
P99 E2E Latency (ms):                    1416788.01
---------------Time to First Token----------------
Mean TTFT (ms):                          422162.05 
Median TTFT (ms):                        443131.76 
P99 TTFT (ms):                           870060.31 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          81.01     
Median TPOT (ms):                        45.88     
P99 TPOT (ms):                           503.28    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           71.93     
Median ITL (ms):                         35.58     
P95 ITL (ms):                            92.47     
P99 ITL (ms):                            148.48    
Max ITL (ms):                            886752.82 
==================================================
```
