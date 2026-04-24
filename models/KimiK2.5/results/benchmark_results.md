
```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2096.63   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6430142   
Request throughput (req/s):              0.73      
Input token throughput (tok/s):          374.40    
Output token throughput (tok/s):         3069.15   
Peak output token throughput (tok/s):    6889.00   
Peak concurrent requests:                517       
Total token throughput (tok/s):          3443.55   
Concurrency:                             430.43    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   587528.64 
Median E2E Latency (ms):                 593909.66 
P90 E2E Latency (ms):                    1048250.42
P99 E2E Latency (ms):                    1205616.90
---------------Time to First Token----------------
Mean TTFT (ms):                          4674.27   
Median TTFT (ms):                        249.73    
P99 TTFT (ms):                           23462.42  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          147.45    
Median TPOT (ms):                        143.86    
P99 TPOT (ms):                           158.83    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           139.36    
Median ITL (ms):                         134.19    
P95 ITL (ms):                            203.88    
P99 ITL (ms):                            569.64    
Max ITL (ms):                            22167.12  
==================================================
```
