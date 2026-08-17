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
Max request concurrency:                 64        
Successful requests:                     192       
Benchmark duration (s):                  3409.86   
Total input tokens:                      196608    
Total input text tokens:                 196608    
Total generated tokens:                  1572864   
Total generated tokens (retokenized):    1572695   
Request throughput (req/s):              0.06      
Input token throughput (tok/s):          57.66     
Output token throughput (tok/s):         461.27    
Peak output token throughput (tok/s):    640.00    
Peak concurrent requests:                67        
Total token throughput (tok/s):          518.93    
Concurrency:                             63.16     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1121618.20
Median E2E Latency (ms):                 890406.87 
P90 E2E Latency (ms):                    1618422.51
P95 E2E Latency (ms):                    1630498.28
P99 E2E Latency (ms):                    1644618.35
---------------Time to First Token----------------
Mean TTFT (ms):                          2598.57   
Median TTFT (ms):                        1016.69   
P90 TTFT (ms):                           8542.95   
P95 TTFT (ms):                           8941.26   
P99 TTFT (ms):                           10006.01  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          136.62    
Median TPOT (ms):                        108.59    
P90 TPOT (ms):                           196.58    
P95 TPOT (ms):                           198.07    
P99 TPOT (ms):                           199.57    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           136.62    
Median ITL (ms):                         108.71    
P90 ITL (ms):                            136.46    
P95 ITL (ms):                            141.59    
P99 ITL (ms):                            154.63    
Max ITL (ms):                            447531.48 
==================================================


```