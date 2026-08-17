```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 128       
Successful requests:                     384       
Benchmark duration (s):                  2857.37   
Total input tokens:                      393216    
Total input text tokens:                 393216    
Total generated tokens:                  3145728   
Total generated tokens (retokenized):    3145043   
Request throughput (req/s):              0.13      
Input token throughput (tok/s):          137.61    
Output token throughput (tok/s):         1100.92   
Peak output token throughput (tok/s):    1280.00   
Peak concurrent requests:                256       
Total token throughput (tok/s):          1238.53   
Concurrency:                             127.97    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   952229.42 
Median E2E Latency (ms):                 953940.19 
P90 E2E Latency (ms):                    965420.78 
P95 E2E Latency (ms):                    965468.83 
P99 E2E Latency (ms):                    965553.84 
---------------Time to First Token----------------
Mean TTFT (ms):                          12142.17  
Median TTFT (ms):                        12571.27  
P90 TTFT (ms):                           21012.92  
P95 TTFT (ms):                           22212.44  
P99 TTFT (ms):                           22261.70  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          114.77    
Median TPOT (ms):                        115.00    
P90 TPOT (ms):                           116.94    
P95 TPOT (ms):                           117.30    
P99 TPOT (ms):                           117.64    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           114.77    
Median ITL (ms):                         112.87    
P90 ITL (ms):                            130.10    
P95 ITL (ms):                            140.12    
P99 ITL (ms):                            173.61    
Max ITL (ms):                            20878.75  

==================================================== 

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 2         
Successful requests:                     2         
Benchmark duration (s):                  45.57     
Total input tokens:                      539       
Total input text tokens:                 539       
Total generated tokens:                  620       
Total generated tokens (retokenized):    620       
Request throughput (req/s):              0.04      
Input token throughput (tok/s):          11.83     
Output token throughput (tok/s):         13.61     
Peak output token throughput (tok/s):    16.00     
Peak concurrent requests:                2         
Total token throughput (tok/s):          25.43     
Concurrency:                             1.78      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   40452.51  
Median E2E Latency (ms):                 40452.51  
P90 E2E Latency (ms):                    44433.70  
P95 E2E Latency (ms):                    44931.35  
P99 E2E Latency (ms):                    45329.47  
---------------Time to First Token----------------
Mean TTFT (ms):                          518.55    
Median TTFT (ms):                        518.55    
P90 TTFT (ms):                           539.00    
P95 TTFT (ms):                           541.55    
P99 TTFT (ms):                           543.60    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          129.25    
Median TPOT (ms):                        129.25    
P90 TPOT (ms):                           129.35    
P95 TPOT (ms):                           129.36    
P99 TPOT (ms):                           129.37    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           129.24    
Median ITL (ms):                         129.36    
P90 ITL (ms):                            130.04    
P95 ITL (ms):                            130.24    
P99 ITL (ms):                            130.56    
Max ITL (ms):                            131.96    
==================================================


```