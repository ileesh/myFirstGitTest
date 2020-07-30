#include <stdio.h>
#include <conio.h>
#include <time.h>

#include "windows.h"

class CHnsElapse
{
public:
	CHnsElapse(void);
	virtual ~CHnsElapse(void);

protected:
	int			m_cnt;
	int			m_cntAvg;

	double		m_total;	// msec
	double		m_avg;		// msec
	double		m_dur;		// msec

	LARGE_INTEGER m_lnFreq;
	LARGE_INTEGER m_lnStart;	// usec
	LARGE_INTEGER m_lnEnd;		// usec
	LARGE_INTEGER m_lnDur;		// usec

public:
	void	init(int nAvg=100);

	double	getAvg() { return m_avg; };
	double	getDur() { return m_dur; };

	void	start();
	void	stop();
};

CHnsElapse::CHnsElapse(void)
{
	init();
}

CHnsElapse::~CHnsElapse(void)
{
}

void CHnsElapse::init(int nAvg)
{
	m_cnt = 0;
	m_cntAvg = nAvg;
	m_total = 0;
	m_avg = 0;
	m_dur = 0;

	memset(&m_lnFreq, 0x00, sizeof(m_lnFreq));
	memset(&m_lnStart, 0x00, sizeof(m_lnStart));
	memset(&m_lnEnd, 0x00, sizeof(m_lnEnd));
	memset(&m_lnDur, 0x00, sizeof(m_lnDur));
}

void CHnsElapse::start()
{
	// usec resolution
	::QueryPerformanceFrequency(&m_lnFreq);
	::QueryPerformanceCounter(&m_lnStart);
}

void CHnsElapse::stop()
{
	// usec resolution
	::QueryPerformanceCounter(&m_lnEnd);
	m_lnDur.QuadPart = (m_lnEnd.QuadPart - m_lnStart.QuadPart) * 1000000 / m_lnFreq.QuadPart;

	if (m_cnt > m_cntAvg) {
		m_cnt = 0;
		m_total = 0;
		m_avg = 0;
	}
	m_cnt++;
	m_dur = m_lnDur.QuadPart / 1000.0;	// msec
	m_total += m_dur;
	m_avg = m_total / m_cnt;
}

#define CUDA	1

__global__ void doSomething(int *p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int idx = (blockDim.x*4)*(blockIdx.y*blockDim.y) + (blockIdx.x * blockDim.x + threadIdx.x)*threadIdx.y;
	p[idx] >>= 1;
}

__global__ void VectorAdd(int *a, int *b, float *c, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float sv = sin((float)a[idx]);
	float cv = cos((float)b[idx]);
	c[idx] = (float)(sv*sv+cv*cv);
}

void NativeVectorAdd(int *a, int *b, float *c, int size)
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int idx = 0; idx < size; idx++) {
		float sv = sin((float)a[idx]);
		float cv = cos((float)b[idx]);
		c[idx] = (float)(sv*sv+cv*cv);
	}
}

int main()
{
	const int size = 65535 * 512;
	const int BufferSize = size*sizeof(int);
	const int FloatBufferSize = size*sizeof(float);

	int *InputA, *InputB;
	float *Result;

	InputA = (int *)malloc(BufferSize);
	InputB = (int *)malloc(BufferSize);
	Result = (float *)malloc(FloatBufferSize);

	int i = 0;

	for(int i = 0; i < size; i++) {
		InputA[i] = i;
		InputB[i] = i;
		Result[i] = 0;
	}

	
#if CUDA
	int *dev_A, *dev_B;
	float *dev_C;
	CHnsElapse et;

	et.init();
	et.start();
	cudaMalloc((void **)&dev_A, BufferSize);
	cudaMalloc((void **)&dev_B, BufferSize);
	cudaMalloc((void **)&dev_C, FloatBufferSize);
	et.stop();
	printf("cudaMalloc time : %f msec\n", et.getDur());

	et.init();
	et.start();
	cudaMemcpy(dev_A, InputA, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, InputB, BufferSize, cudaMemcpyHostToDevice);
	et.stop();
	printf("cudaMemcpy time to send data: %f msec\n", et.getDur());
	
	et.init();
	et.start();
	VectorAdd<<<65535, 512>>>(dev_A, dev_B, dev_C, size);
	et.stop();
	printf("running time : %f msec\n", et.getDur());

	et.init();
	et.start();
	cudaMemcpy(Result, dev_C, FloatBufferSize, cudaMemcpyDeviceToHost);
	et.stop();
	printf("cudaMemcpy time to receive the result: %f msec\n", et.getDur());
#else
	CHnsElapse et;
	et.init();
	et.start();
	NativeVectorAdd(InputA, InputB, Result, size);
	et.stop();
	printf("running time : %f msec\n", et.getDur());

#endif
	
	for(i = 0; i < 5; i++)
		printf("[%d] = %f\n", i, Result[i]);
	printf("....\n");
	for(i = size-5; i < size; i++)
		printf("[%d] = %f\n", i, Result[i]);

#if CUDA
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
#endif

	free(InputA);
	free(InputB);
	free(Result);

	getch();
}