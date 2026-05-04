```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    inf       
Max request concurrency:                 96        
Successful requests:                     200       
Benchmark duration (s):                  156.24    
Total input tokens:                      416595    
Total input text tokens:                 8272      
Total input vision tokens:               408323    
Total generated tokens:                  406325    
Total generated tokens (retokenized):    390717    
Request throughput (req/s):              1.28      
Input token throughput (tok/s):          2666.41   
Output token throughput (tok/s):         2600.67   
Peak output token throughput (tok/s):    4603.00   
Peak concurrent requests:                100       
Total token throughput (tok/s):          5267.08   
Concurrency:                             77.29     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   60380.75  
Median E2E Latency (ms):                 56476.74  
P90 E2E Latency (ms):                    109825.80 
P99 E2E Latency (ms):                    124183.10 
---------------Time to First Token----------------
Mean TTFT (ms):                          5509.54   
Median TTFT (ms):                        454.43    
P99 TTFT (ms):                           16798.21  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.47     
Median TPOT (ms):                        29.14     
P99 TPOT (ms):                           85.47     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           27.94     
Median ITL (ms):                         22.18     
P95 ITL (ms):                            46.72     
P99 ITL (ms):                            243.95    
Max ITL (ms):                            14422.16  
==================================================
```
