#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>

#include "StopWatch.h"

FILE *in, *out;
unsigned short *bin;
unsigned char *bout;
int w, h;
int insize, outsize;
CStopWatch tm;

#define clip(var) ((var>=255)?255:(var<=0)?0:var)

void YUV420toRGB(unsigned short* pYuvbuff, unsigned char* pRgbbuff, int width, int height, int nBits)
{
	int i, j;
	int c, d, e;
	unsigned char* y, *u, *v;
	unsigned char* cur = pRgbbuff;
	unsigned char* pBuff;
	int size;

	if(!pYuvbuff || !pRgbbuff)
		return;

	size = (width*height*3)/2;
	pBuff = new unsigned char[size];
	if(!pBuff) {
		printf("Memory alloc fail...\n");
		return;
	}

	if(nBits > 8)
	{
		for(i = 0; i < size; i++)
		{
			pBuff[i] = (pYuvbuff[i] >> (nBits-8));
		}
	}
	else
	{
		memcpy(pBuff, (unsigned char *)pYuvbuff, size);
	}

	y = (unsigned char *)pBuff;
	u = (unsigned char *)pBuff + (width * height);
	v = (unsigned char *)pBuff + (width * height) + (width * height)/4;

	for( j = 0 ; j < height ; j++ ){
		for( i = 0 ; i < width ; i++ ){
			c = y[j*width+i] - 16;
			d = u[(j>>1)*(width>>1)+(i>>1)] - 128;
			e = v[(j>>1)*(width>>1)+(i>>1)] - 128;
			(*cur) = clip(( 298 * c           + 409 * e + 128) >> 8);cur++;
			(*cur) = clip(( 298 * c - 100 * d - 208 * e + 128) >> 8);cur++;
			(*cur) = clip(( 298 * c + 516 * d           + 128) >> 8);cur++;
		}
	}

	delete pBuff;
}

int main(int argc, char **argv)
{
	if(argc < 5) {
		printf("Usage : YUV2RGB width height in_yuv out_rgb\n");
		return -1;
	}

	w = atoi(argv[1]);
	h = atoi(argv[2]);
	insize = (w*h*3/2);
	outsize = w*h*3;
	bin = (unsigned short *)malloc(insize);
	bout = (unsigned char *)malloc(outsize);
	memset(bout, 0x00, outsize);
	in = fopen(argv[3], "rb");
	out = fopen(argv[4], "wb");

	fread(bin, insize, sizeof(unsigned char), in);

	tm.Start();
	YUV420toRGB(bin, bout, w, h, 8);
	tm.End();
	printf("elapsed time to YUV2RGB... : %f msec\n", tm.GetDurationMilliSecond());
	fwrite(bout, outsize, sizeof(unsigned char), out);

	fclose(in);
	fclose(out);
	free(bin);
	free(bout);

	return 0;
}
