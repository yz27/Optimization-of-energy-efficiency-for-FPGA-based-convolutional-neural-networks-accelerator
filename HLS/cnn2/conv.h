#ifndef __conv_H
#define	__conv_H
//#include "common.h"
#include<ap_int.h>
typedef float  data_type;
typedef ap_int<8> int8;
#define N 32
#define N2 28
#define convbia 2
#define N3 14
#define N4 10
#define N5 5
#define M 5
#define Depth 6
#define Depth1 16
#define Depth2 120
#define insize1 120
#define outsize1 84
#define insize2 84
#define outsize2 10



void Convolution_layer1(data_type imagein_conv1[32][32], data_type imageout_conv1[Depth][N2][N2]);
void Convolution_layer2(data_type imagein_conv2[Depth][N3][N3],data_type imageout_conv2[Depth1][N4][N4]);
void Convolution_layer3(data_type imagein_conv3[Depth1][N5][N5],data_type imageout_conv3[Depth2]);

void MaxPool_layer1(data_type imagein[Depth][N2][N2],data_type imageout[Depth][N3][N3]);
void MaxPool_layer2(data_type imagein_maxpool2[Depth1][N4][N4],data_type imageout2[Depth1][N5][N5]);


void fullconnected1(data_type imagein1[insize1],data_type imageout1[outsize1]);
void fullconnected2(data_type imagein2[insize2],data_type imageout2[outsize2]);


int8 cnn(data_type imagein[1024]);
#endif

