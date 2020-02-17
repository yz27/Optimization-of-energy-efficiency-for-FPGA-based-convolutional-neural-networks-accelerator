#ifndef __maxpool
#define __maxpool

#include "common.h"

void MaxPool_layer1(data_type imagein[Depth][N2][N2],data_type imageout[Depth][N3][N3]);
void MaxPool_layer2(data_type imagein_maxpool2[Depth1][N4][N4],data_type imageout2[Depth1][N5][N5]);
#endif
