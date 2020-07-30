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
#define MSIZE	2048

void NativeMatrixMult(int *M, int *N, int *P, int width)
{
	int row, col;
	int index = 0;
	int destindex = 0;

	for(row = 0; row < width; row++) {
		for(col = 0; col < width; col++) {
			destindex = row*width+col;
			for(index = 0; index < width; index++) {
				P[destindex] += M[row*width+index]*N[index*width+col];
			}
		}
	}
}

__global__ void MatrixMult(int *M, int *N, int *P, int width)
{
	int tid, tx, ty;

	tx = blockIdx.x*blockDim.x + threadIdx.x;
	ty = blockIdx.y*blockDim.y + threadIdx.y;
	tid = ty*width + tx;
	int Pv = 0, Mv = 0, Nv = 0;

	for(int i = 0; i < width; i++) {
		Mv = M[ty*width+i];
		Nv = N[i*width+tx];
		Pv += Mv * Nv;
	}

	P[tid] = Pv;
}

int  devcheck(int gpudevice) 
{ 
	int device_count=0; 
	int device;  // used with  cudaGetDevice() to verify cudaSetDevice() 

	// get the number of non-emulation devices  detected 
	cudaGetDeviceCount( &device_count); 
	if (gpudevice > device_count) 
	{ 
		printf("gpudevice >=  device_count ... exiting\n"); 
		exit(1); 
	} 
	cudaError_t cudareturn; 
	cudaDeviceProp deviceProp; 

	// cudaGetDeviceProperties() is also  demonstrated in the deviceQuery/ example
	// of the sdk projects directory 
	cudaGetDeviceProperties(&deviceProp,  gpudevice); 
	printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n", 
		deviceProp.major, deviceProp.minor); 

	if (deviceProp.major > 999) 
	{ 
		printf("warning, CUDA Device  Emulation (CPU) detected, exiting\n"); 
		exit(1); 
	} 

	// choose a cuda device for kernel  execution 
	cudareturn=cudaSetDevice(gpudevice); 
	if (cudareturn == cudaErrorInvalidDevice) 
	{ 
		perror("cudaSetDevice returned  cudaErrorInvalidDevice"); 
	} 
	else 
	{ 
		// double check that device was  properly selected 
		cudaGetDevice(&device); 
		printf("cudaGetDevice()=%d\n",device);
	}
	return 0;
} 

