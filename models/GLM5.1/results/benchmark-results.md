# GLM-5.1 Benchmark Results on GCP G4

This file contains the detailed benchmark logs for various configurations of GLM-5.1 on RTX PRO 6000 GPUs.

## FP8 (2-Node Setup)

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

## NVFP4 (1-Node Setup)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  4317.82   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6425523   
Request throughput (req/s):              0.36      
Input token throughput (tok/s):          181.80    
Output token throughput (tok/s):         1490.31   
Peak output token throughput (tok/s):    734.00    
Peak concurrent requests:                516       
Total token throughput (tok/s):          1672.11   
Concurrency:                             423.71    
Accept length:                           3.25      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1191094.40
Median E2E Latency (ms):                 1230999.34
P90 E2E Latency (ms):                    1543889.47
P99 E2E Latency (ms):                    2479451.05
---------------Time to First Token----------------
Mean TTFT (ms):                          900508.65 
Median TTFT (ms):                        1064328.97
P99 TTFT (ms):                           1186020.43
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          73.82     
Median TPOT (ms):                        41.17     
P99 TPOT (ms):                           543.11    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           69.46     
Median ITL (ms):                         34.33     
P95 ITL (ms):                            78.42     
P99 ITL (ms):                            142.61    
Max ITL (ms):                            1176618.49
==================================================
```

## NVFP4 (2-Node Setup)

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2092.07   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6423233   
Request throughput (req/s):              0.73      
Input token throughput (tok/s):          375.21    
Output token throughput (tok/s):         3075.85   
Peak output token throughput (tok/s):    4606.00   
Peak concurrent requests:                516       
Total token throughput (tok/s):          3451.06   
Concurrency:                             407.12    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   554508.10 
Median E2E Latency (ms):                 557166.40 
P90 E2E Latency (ms):                    974727.07 
P99 E2E Latency (ms):                    1113768.94
---------------Time to First Token----------------
Mean TTFT (ms):                          5629.43   
Median TTFT (ms):                        331.11    
P99 TTFT (ms):                           29023.03  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          141.36    
Median TPOT (ms):                        134.68    
P99 TPOT (ms):                           156.93    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           131.28    
Median ITL (ms):                         124.22    
P95 ITL (ms):                            203.11    
P99 ITL (ms):                            287.09    
Max ITL (ms):                            28242.56  
==================================================
```
