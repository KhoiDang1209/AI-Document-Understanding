# DocIntel KIE Benchmark — LayoutLMv3 (fp32 vs ONNX vs INT8)

| Config | F1 | Precision | Recall | Accuracy | p50 (ms) | p95 (ms) | Throughput (doc/s) | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| torch-fp32 | 0.8449 | 0.8343 | 0.8557 | 0.8700 | 1934.2 | 2379.9 | 0.50 | 480.5 |
| onnx-fp32 | 0.8449 | 0.8343 | 0.8557 | 0.8700 | 1214.7 | 1401.4 | 0.78 | 480.8 |
| onnx-int8 | 0.8315 | 0.8177 | 0.8457 | 0.8651 | 650.2 | 852.0 | 1.46 | 121.4 |

![benchmark_latency_p95](benchmark_latency_p95.png)
![benchmark_f1](benchmark_f1.png)
![benchmark_size_mb](benchmark_size_mb.png)
