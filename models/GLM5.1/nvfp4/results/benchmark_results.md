Benchmark_results for glm5.1 model from lukealonso/GLM-5.1-NVFP4. Since as of April 26.2026. There is no official NVIDIA HF checkpoint.


```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     512       
Benchmark duration (s):                  1863.46   
Total input tokens:                      258921    
Total input text tokens:                 258921    
Total generated tokens:                  2054943   
Total generated tokens (retokenized):    2049196   
Request throughput (req/s):              0.27      
Input token throughput (tok/s):          138.95    
Output token throughput (tok/s):         1102.76   
Peak output token throughput (tok/s):    630.00    
Peak concurrent requests:                512       
Total token throughput (tok/s):          1241.70   
Concurrency:                             262.68    
Accept length:                           3.39      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   956058.90 
Median E2E Latency (ms):                 922631.88 
P90 E2E Latency (ms):                    1688084.74
P99 E2E Latency (ms):                    1843167.86
---------------Time to First Token----------------
Mean TTFT (ms):                          619332.93 
Median TTFT (ms):                        590304.91 
P99 TTFT (ms):                           1410424.95
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          96.29     
Median TPOT (ms):                        49.16     
P99 TPOT (ms):                           817.91    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           84.13     
Median ITL (ms):                         40.58     
P95 ITL (ms):                            88.96     
P99 ITL (ms):                            167.76    
Max ITL (ms):                            1288981.91
==================================================
```
