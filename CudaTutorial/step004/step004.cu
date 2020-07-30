#include <stdio.h>
#include <conio.h>

#define MAX_SHAREDSIZE	2048

__global__ void LoadStoreViaSharedMemory(int *In, int *Out)
{
#if 1
	int LoadStoreSize = MAX_SHAREDSIZE/blockDim.x;
	int beginIndex = threadIdx.x * LoadStoreSize;
	int endIndex = beginIndex + LoadStoreSize;

	// 공유 메모리 할당
	__shared__ int SharedMemory[MAX_SHAREDSIZE];
	int i;

	for(i = beginIndex; i < endIndex; i++)
		SharedMemory[i] = In[i];

	__syncthreads();

	for(i = beginIndex; i < endIndex; i++)
		Out[i] = SharedMemory[i];

	__syncthreads();
#else
	__shared__ int SharedMemory[MAX_SHAREDSIZE];

	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	SharedMemory[idx] = In[idx];
	Out[idx] = SharedMemory[idx];
#endif
}

int main()
{
	int size = MAX_SHAREDSIZE;
	int BufferSize = size*sizeof(int);

	int *In, *Out;
	In = (int *)malloc(BufferSize);
	Out = (int *)malloc(BufferSize);

	int i = 0;

	for(i = 0; i < size; i++) {
		In[i] = i;
		Out[i] = 0;
	}

	int *devIn, *devOut;
	cudaMalloc((void **)&devIn, BufferSize);
	cudaMalloc((void **)&devOut, BufferSize);
	
	cudaMemcpy(devIn, In, BufferSize, cudaMemcpyHostToDevice);

	//LoadStoreViaSharedMemory<<<32, 64>>>(devIn, devOut);
	LoadStoreViaSharedMemory<<<1, 512>>>(devIn, devOut);

	cudaMemcpy(Out, devOut, BufferSize, cudaMemcpyDeviceToHost);

	for(i = 0; i < 5; i++) 
		printf("%04d\n", Out[i]);

	printf("......\n");
	for(i = size-5; i < size; i++)
		printf("%04d\n", Out[i]);

	cudaFree(devIn);
	cudaFree(devOut);

	free(In);
	free(Out);

	getch();
}

