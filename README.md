# 编译并运行代码
## 编译

1. 编译原始CPU代码
nvcc -std=c++11 -o nbody 01-nbody.cu
2. 编译nbody_parallel.cu，这是基础的GPU加速版本
nvcc -arch=sm_80 -o nbody_GPU_basic nbody_parallel.cu
3. 编译nbody_shared.cu, 这是做了内存优化的GPU加速版本
nvcc -arch=sm_80 -o nbody_GPU_shared nbody_shared.cu

## 运行效果

```
./nbody 4096 Bodies: average 0.081 Billion Interactions / second
./nbody_GPU_basic 4096 Bodies: average 32.320 Billion Interactions / second
./nbody_GPU_shared 4096 Bodies: average 197.147 Billion Interactions / second
```

## 性能分析——仅简要回答，详细分析在下一部分

- 通过launch kernel function，并行计算每个天体受到的引力，并行更新每个天体的坐标，nbody_parallel.cu比01-nbody.cu执行快
- `nbody_shared.cu`比`nbody_parallel.cu`快大约六倍，主要原因是：
  1. 提高并行度。先前并行的方式是每个线程计算一个天体与其他所有天体的相互作用，现在在计算每个天体时，使用`BLOCK_STRIDE`个进程并行计算该天体与其他所有天体的相互作用。
  2. 使用shared memory存取速度更快
  3. 使用shared memory减少访存次数
- `nbody_shared.cu`比`nbody_parallel.cu`快，主要因为修改了`bodyForce`这个kernel function的实现
- `nbody_shared.cu`比`01-nbody.cu`执行快大约1434倍，主要原因是充分利用gpu并行计算的能力，同时用`shared memory`优化访存速度
# 分析代码与加速原因

## 01-nbody.cu-N体模拟器的运算

- 每个天体的运动状态，是通过坐标和速度，`(x, y, z, vx, vy, vz)`来描述
- 考虑天体之间的引力相互作用，天体运行速度是不断变化的，取微元时间`dt`，在此期间近似为匀速直线运动
    ```c
    for (int i = 0; i < nBodies; i++)
        { // integrate position
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    ```
- 在每个微元之间后，更新每个天体的运动速度，近似为匀加速运动。具体方法是，将受力分解为`(x, y, z)`三个方向，计算每个天体受其他天体作用的合力，进而得到每个方向的加速度，以更新每个方向的速度`(vx, vy, vz)`
    ```c
    void bodyForce(Body *p, float dt, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            // 计算每个天体在(x, y, z)方向，受到其他天体作用的合力
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;

            for (int j = 0; j < n; j++)
            {
                // 计算天体之间的距离
                float dx = p[j].x - p[i].x;
                float dy = p[j].y - p[i].y;
                float dz = p[j].z - p[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                // 平方反比率
                // 三次方是为了后面计算cos，做力的分解
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                // 分解到(x, y, z)三个方向
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }

            // 近似为匀加速，更新天体运行速度(vx, vy, vz)
            p[i].vx += dt * Fx;
            p[i].vy += dt * Fy;
            p[i].vz += dt * Fz;
        }
    }
    ```
- 在每一个微元运算时，先调用`bodyForce`更新每个天体的速度，然后更新每个天体的坐标
    ```c
    # 更新每个天体的速度
    bodyForce(p, dt, nBodies); // compute interbody forces
    # 更新每个天体的坐标
    for (int i = 0; i < nBodies; i++)
        { // integrate position
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    ```
- **原始版本是在cpu上串行计算，通过循环遍历每一个天体的方式，更新其速度和坐标**
  
## nbody_parallel.cu-在gpu上并行计算

### 代码思路分析
- **在gpu上并行运算，`bodyForce`中每个线程负责计算和更新一个天体的速度，`integrate_position`中每个线程负责更新一个天体的坐标**
    ```c
    // kernel function并行处理每个天体的方法，bodyForce的思路与此一致
    __global__ void integrate_position(Body *p, float dt, int n)
    {
        // 通过当前thread的编号和其所在的block的编号，索引到该线程负责计算的天体的编号
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        // 更新该天体的坐标
        if (i < n)
        {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }
    ```
- `cudaMallocManaged`：分配一块统一内存空间，指针指向这块内存，可以在主机和设备之间共享
    ```c
    // 计算需要分配多少字节的空间
    int bytes = nBodies * sizeof(Body);
    float *buf;
    // 在cpu上分配空间
    // buf = (float *)malloc(bytes);
    // 在gpu上分配空间
    cudaMallocManaged(&buf, bytes);
    // 强制类型转换，将内存区域解释为Body类型的数据
    Body *p = (Body *)buf;
    ```
- launch kernel的参数中，因为一个warp有32个线程，所以`BLOCK_SIZE`应该为32的倍数。`numberOfBlocks`取决于天体的数量，在后续Nsight分析的时候，将探究`numberOfBlocks`对性能的影响
    ```c
    #define BLOCK_SIZE 32
    size_t threadsPerBlock = BLOCK_SIZE;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
    ```

- Kernel在默认流中顺序执行，所以会先计算完每个天体的速度，再开始更新每个天体的位置
    ```c
    bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces
    integrate_position<<<numberOfBlocks,threadsPerBlock>>>(p,dt,nBodies);
    ```

### Nsight分析

## nbody_shared.cu

### 代码思路分析

- 用`BLOCK_STRIDE`个thread，并行计算一个天体与其他天体的相互作用，相比先前一个进程计算一个天体与其他天体的相互作用，并行度提高
  - 启动`numberOfBlocks * BLOCK_STRIDE`个block，在kernel中`BLOCK_STRIDE`个进程索引到同一个天体
    ```c
    // launch kernel的参数中，block的数量变为原来的BLOCK_STRIDE倍，总进程数变为原来的BLOCK_STRIDE倍(threadsPerBlock未变)，用于并行计算一个天体与其他天体的相互作用
    bodyForce<<<numberOfBlocks * BLOCK_STRIDE, threadsPerBlock>>>(d_p, dt, nBodies); // compute interbody forces

    __global__ void bodyForce(Body *p, float dt, int n)
    {
        // 先前每个进程计算一个天体与其他天体的相互作用
        // int i = threadIdx.x + blockIdx.x * blockDim.x;
        // 计算要处理的数据index
        // 现在BLOCK_STRIDE个进程索引到同一个天体
        int i = threadIdx.x + (int)(blockIdx.x / BLOCK_STRIDE) * blockDim.x;
        ...
    }
    ```
  - 每个天体要计算与其他n-1个天体的相互作用，在代码中体现为与n个天体进行计算（与自身的力为零）。将n个`Body`的数据，分成多个`page`，每个`page`有`BLOCK_STRIDE`个`block`。
  - 在并行计算的时候，索引到某个天体的有`BLOCK_STRIDE`个进程，其中每个进程负责计算`page`中第`blockIdx.x % BLOCK_STRIDE`个`block`中的天体数据，然后不断循环，跳转到下一个`page`中同样位置的`block`的天体数据，直到遍历完所有天体
  - 因为是并行计算所以在最后累加的时候，需要采取原子加，以避免竞争的现象
    ```c

    __global__ void bodyForce(Body *p, float dt, int n)
    {
        ...
        // 此块对应要处理的数据块的起始位置
        // 当前进程负责为索引到的天体，计算每个page中，第start_block块中包含的所有天体，对当前天体的相互作用
        int start_block = blockIdx.x % BLOCK_STRIDE;
        ...
        if (i < n)
        {
            int cycle_times = n / BLOCK_SIZE;
            // 当前线程计算的天体
            Body ptemp = p[i];
            ...
            float dx, dy, dz, distSqr, invDist, invDist3;
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;
            // 这里的cycle_times 在已知块大小时使用常数性能会高一些
            // rollout the pages
            for (int block_num = start_block; block_num < cycle_times; block_num += BLOCK_STRIDE)
            {
                ...
                // spos中存储当前page，当前进程负责的block中所有天体的数据
                // 计算其中每一个天体与当前天体的相互作用
                for (int j = 0; j < BLOCK_SIZE; j++)
                {
                    dx = spos[j].x - ptemp.x;
                    dy = spos[j].y - ptemp.y;
                    dz = spos[j].z - ptemp.z;
                    distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    invDist = rsqrtf(distSqr);
                    invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
                ...
            }
            // 当前天体与其他天体的相互作用，被分到BLOCK_SIZE个进程计算，在每个进程计算完毕后，进程间的要累加起来计算合力
            // 由于块之间不同步，原子加保证正确性，避免竞争现象
            atomicAdd(&p[i].vx, dt * Fx);
            atomicAdd(&p[i].vy, dt * Fy);
            atomicAdd(&p[i].vz, dt * Fz);
            // p[i].vx += dt * Fx;
            // p[i].vy += dt * Fy;
            // p[i].vz += dt * Fz;
        }
    }
    ```

- 使用`shared memory`进行优化
  

# 参考
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

https://docs.nvidia.com/cuda/cuda-c-programming-guide/