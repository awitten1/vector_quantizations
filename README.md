# vector_quantizations

## Build

```sh
cmake -S . -B build -G Ninja
cmake --build build
./build/vq_cli       # benchmark: compression, error, recall@10
./build/pq_example   # small walkthrough of train / encode / decode / search
```

## Test

```sh
ctest --test-dir build --output-on-failure
```
