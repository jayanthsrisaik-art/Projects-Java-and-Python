# deltasync benchmark (file size 262144 bytes, seeds [1, 2, 3])

Mean total wire bytes (receiver->sender + sender->receiver). Lower is better.

| workload | whole-file | aligned-blocks | fixed-dedupe | cdc-dedupe | rsync-delta |
|---|---:|---:|---:|---:|---:|
| identical | 262,144 | 1,160 | 1,024 | **917** | 4,557 |
| append-4k | 266,240 | 5,258 | **5,140** | 9,303 | 9,000 |
| prepend-1b | 262,145 | 263,306 | 263,317 | 10,821 | **4,560** |
| insert-1k-middle | 263,168 | 133,940 | 135,936 | 23,917 | **6,291** |
| scattered-20-edits | 262,144 | 41,437 | 134,892 | 162,316 | **18,735** |
| delete-10k | 251,904 | 103,555 | 105,508 | 10,306 | **5,294** |
| move-32k-section | 262,144 | 185,480 | 50,200 | 56,150 | **6,686** |
| full-rewrite | **262,144** | 263,304 | 263,296 | 263,183 | 266,700 |

All reconstructions byte-identical: **True**

Mean seconds per sync (pure Python, includes signature + delta + patch):

| workload | whole-file | aligned-blocks | fixed-dedupe | cdc-dedupe | rsync-delta |
|---|---:|---:|---:|---:|---:|
| identical | 0.000 | 0.001 | 0.001 | 0.042 | 0.025 |
| append-4k | 0.000 | 0.001 | 0.001 | 0.036 | 0.027 |
| prepend-1b | 0.000 | 0.001 | 0.001 | 0.042 | 0.029 |
| insert-1k-middle | 0.000 | 0.001 | 0.001 | 0.039 | 0.028 |
| scattered-20-edits | 0.000 | 0.001 | 0.001 | 0.046 | 0.039 |
| delete-10k | 0.000 | 0.001 | 0.001 | 0.038 | 0.029 |
| move-32k-section | 0.000 | 0.001 | 0.001 | 0.039 | 0.029 |
| full-rewrite | 0.000 | 0.001 | 0.001 | 0.040 | 0.070 |

## rsync-delta block-size sweep (scattered-20-edits)

| block size | signature up | delta down | total |
|---:|---:|---:|---:|
| 128 | 24,587 | 2,960 | 27,547 |
| 256 | 12,299 | 5,520 | 17,819 |
| 512 | 6,155 | 10,634 | 16,789 |
| 700 | 4,511 | 14,224 | 18,735 |
| 2048 | 1,547 | 40,480 | 42,027 |
| 8192 | 395 | 133,929 | 134,324 |
