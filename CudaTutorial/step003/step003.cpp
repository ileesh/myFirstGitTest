#include <conio.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include "step003.h"

unsigned char *In, *Out;

int main()
{
	const int size = WIDTH*HEIGHT;
	const int BufferSize = size*sizeof(unsigned char);

	In = (unsigned char *)malloc(BufferSize);
	Out = (unsigned char *)malloc(BufferSize);

	FILE *fp = fopen("lena4096.raw", "rb");
	fread(In, sizeof(unsigned char), size, fp);
	fclose(fp);

	//memcpy(Out, In, BufferSize);
	memset(Out, 0x00, BufferSize);

	smooth(In, Out);

	fp = fopen("cuda_lena4096_out.raw", "wb");
	fwrite(Out, sizeof(unsigned char), size, fp);
	fclose(fp);

	free(In);
	free(Out);


	getch();
}