#include <stdio.h>
#include <conio.h>
#include <time.h>

#include "windows.h"
#include "step003.h"

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

//void Native2dConvolution(int *In, float *Out, int width, int height, int fsize)
void NativeImageSmooth(unsigned char *In, unsigned char *Out, int width, int height, int fsize)
{
	int row, col, frow, fcol;
	int indent = fsize/2;
	int destindex = 0;
	float tmp;

	for(row = indent; row < height-indent; row++) {
		for(col = indent; col < width-indent; col++) {
			destindex = row*width+col;
			tmp = 0.0f;
			for(frow = -fsize/2; frow <= fsize/2; frow++) {
				for(fcol = -fsize/2; fcol <= fsize/2; fcol++) {
					tmp += (float)In[(row+frow)*width+(col+fcol)];
				}
			}
			tmp /= (fsize*fsize);	// average
			Out[destindex] = (unsigned char)tmp;
		}
	}
}

__global__ void CudaImageSmooth(unsigned char *In, unsigned char *Out, int width, int height, int fsize)
{
	int row, col, destIndex;

	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	destIndex = row*width + col;
	int frow, fcol;
	float tmp = 0.0;

	if(col < fsize/2 || col > width-fsize/2 || row < fsize-2 || row > width-fsize/2) {
		Out[destIndex] = 0;
	} else {
		for(frow = -fsize/2; frow <= fsize/2; frow++) {
			for(fcol = -fsize/2; fcol <= fsize/2; fcol++) {
				tmp += (float)In[(row+frow)*width+(col+fcol)];
			}
		}
		tmp /= (fsize*fsize);	// average
		Out[destIndex] = (unsigned char)tmp;
	}
}

int smooth(unsigned char *In, unsigned char *Out)
{
	const int size = WIDTH*HEIGHT;
	const int BufferSize = size*sizeof(unsigned char);

	unsigned char *dIn, *dOut;
#if CUDA
	CHnsElapse et;

	et.init();
	et.start();
	cudaMalloc((void **)&dIn, BufferSize);
	cudaMalloc((void **)&dOut, BufferSize);
	et.stop();
	printf("cudaMalloc time : %f msec\n", et.getDur());
	//cudaMalloc((void **)&dOut, BufferSize);


	et.init();
	et.start();
	cudaMemcpy(dIn, In, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dOut, Out, BufferSize, cudaMemcpyHostToDevice);
	et.stop();
	printf("cudaMemcpy time to send data: %f msec\n", et.getDur());
	
	et.init();
	et.start();
	dim3 grid(128, 256, 1);
	dim3 block(32, 16, 1);
	CudaImageSmooth<<<grid, block>>>(dIn, dOut, WIDTH, HEIGHT, 9);
	cudaThreadSynchronize();
	et.stop();
	printf("running time : %f msec\n", et.getDur());

	et.init();
	et.start();
	cudaMemcpy(Out, dOut, BufferSize, cudaMemcpyDeviceToHost);
	et.stop();
	printf("cudaMemcpy time to receive the result: %f msec\n", et.getDur());
#else
	CHnsElapse et;
	et.init();
	et.start();
	NativeImageSmooth(In, Out, WIDTH, HEIGHT, 9);
	et.stop();
	printf("running time : %f msec\n", et.getDur());

#endif

#if CUDA
	cudaFree(dIn);
	cudaFree(dOut);
#endif

	return 1;
}