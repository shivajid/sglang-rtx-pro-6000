
These are the benchmark results running DSv3.2 on 1 node G4/8GPU (g4-standard-384)  using `nvidia/DeepSeek-V3.2-NVFP4` model.

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     512       
Benchmark duration (s):                  768.11    
Total input tokens:                      258921    
Total input text tokens:                 258921    
Total generated tokens:                  2054943   
Total generated tokens (retokenized):    2033267   
Request throughput (req/s):              0.67      
Input token throughput (tok/s):          337.09    
Output token throughput (tok/s):         2675.33   
Peak output token throughput (tok/s):    2046.00   
Peak concurrent requests:                512       
Total token throughput (tok/s):          3012.42   
Concurrency:                             270.66    
Accept length:                           2.87      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   406045.84 
Median E2E Latency (ms):                 413006.85 
P90 E2E Latency (ms):                    685542.08 
P99 E2E Latency (ms):                    739372.68 
---------------Time to First Token----------------
Mean TTFT (ms):                          18660.63  
Median TTFT (ms):                        18680.89  
P99 TTFT (ms):                           31312.05  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          106.03    
Median TPOT (ms):                        98.12     
P99 TPOT (ms):                           214.60    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           96.53     
Median ITL (ms):                         88.08     
P95 ITL (ms):                            142.49    
P99 ITL (ms):                            257.79    
Max ITL (ms):                            28328.48  
==================================================
```
