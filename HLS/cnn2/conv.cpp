#include "conv.h"
//#include "maxpool.h"
//#include "common.h"
#define Max(A,B) ((A>B)?A:B)

const data_type Wconv1[150] = {
#include "W_conv1.h"
};

const data_type Wconv2[2400] = {
#include "W_conv2.h"
};
const data_type Wconv3[48000] = {
#include "W_conv3.h"
};


const data_type bias_conv1[6] = {
#include "b_conv1.h"
};

const data_type bias_conv2[16] = {
#include "b_conv2.h"
};

const data_type bias_conv3[120] = {
#include "b_conv3.h"
};
const data_type W_fc1[10080] = {
#include "W_fc1.h"
};
const data_type W_fc2[840] = {
#include "W_fc2.h"
};
const data_type bias1[84] = {
#include "b_fc1.h"
};
const data_type bias2[10] = {
#include "b_fc2.h"
};


void Convolution_layer1(data_type imagein_conv1[32][32], data_type imageout_conv1[Depth][N2][N2])
{
#pragma HLS ARRAY_PARTITION variable=imagein_conv1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv1 complete dim=2
	int i,j,m,n,p,q;

	data_type W1[1][Depth][M][M];
#pragma HLS ARRAY_PARTITION variable=W1 complete dim=2

	data_type imagein_1[28][28][5][5];
#pragma HLS ARRAY_PARTITION variable=imagein_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imagein_1 complete dim=3
	for(j=0;j<Depth;j++){
		for(m=0;m<M;m++){
			for(n=0;n<M;n++){
				W1[0][j][m][n]=Wconv1[j+m*30+n*6];
			}
		}
	}
		for(q=0;q<M;q++){
				for(n=0;n<N2;n++){
#pragma HLS PIPELINE
					for(m=0;m<N2;m++){
						for(p=0;p<M;p++){
						imagein_1[m][n][p][q]=imagein_conv1[m+p][n+q];

					}

				}}}

	Convolution_layer1_label4:for(p=0;p<M;p++){
		Convolution_layer1_label5:for(q=0;q<M;q++){
				Convolution_layer1_label2:for(n=0;n<N2;n++){
#pragma HLS PIPELINE II=4
					Convolution_layer1_label1:for(m=0;m<N2;m++){
								Convolution_layer1_label0:for(j=0;j<Depth;j++){
									imageout_conv1[j][m][n] +=(imagein_1[m][n][p][q]*W1[0][j][p][q]);

								}

							}
					}
				}
			}


	for(n=0;n<N2;n++){
#pragma HLS PIPELINE
	for(j=0;j<Depth;j++){
		for(m=0;m<N2;m++){
				imageout_conv1[j][m][n]=Max(imageout_conv1[j][m][n],0);
			}
		}
	}


}

void MaxPool_layer1(data_type imagein[Depth][N2][N2],data_type imageout[Depth][N3][N3])
{
#pragma HLS ARRAY_PARTITION variable=imagein complete dim=1
#pragma HLS ARRAY_PARTITION variable=imagein complete dim=2
#pragma HLS ARRAY_PARTITION variable=imageout complete dim=1

	MaxPool_layer1_label2:for(int j=0;j<N3;j++)
	{
		for(int k=0;k<N3;k++)
		{
#pragma HLS PIPELINE
			MaxPool_layer1_label1:for(int i=0;i<Depth;i++)
			{
				imageout[i][j][k]=Max( Max( imagein[i][2*j][2*k], imagein[i][2*j][2*k+1]) , Max( imagein[i][2*j+1][2*k], imagein[i][2*j+1][2*k+1]) );//find the max of the pool
			}
		}
	}

}

void Convolution_layer2(data_type imagein_conv2[Depth][N3][N3],data_type imageout_conv2[Depth1][N4][N4])
{

#pragma HLS ARRAY_PARTITION variable=imagein_conv2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Wconv2 cyclic factor=16 dim=1
	int i,j,m,n,p,q,k;
	data_type W2[Depth][16][M][M];
	data_type W22[Depth][8][M][M];
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W22 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W22 complete dim=2

	for(m=0;m<M;m++){
		for(n=0;n<M;n++){
				for(i=0;i<Depth;i++){
#pragma HLS PIPELINE
					for(j=0;j<Depth1;j++){
							W2[i][j][m][n]=Wconv2[i*8+j+m*240+n*48+150*1200];
				}
			}
		}
	}



	Convolution_layer2_label4:for(p=0;p<M;p++){
		Convolution_layer2_label5:for(q=0;q<M;q++){
			Convolution_layer2_label1:for(m=0;m<N4;m++){
				Convolution_layer2_label2:for(n=0;n<N4;n++){
#pragma HLS PIPELINE II=2
					Convolution_layer2_label3:for(i=0;i<Depth;i++){
								Convolution_layer2_label0:for(j=0;j<Depth1;j++){
									imageout_conv2[j][m][n] +=(imagein_conv2[i][m+p][n+q]*W2[i][j][p][q]);
	}}}}}}


		for(j=0;j<N4;j++){
			for(m=0;m<N4;m++){
#pragma HLS PIPELINE
				for(i=0;i<Depth1;i++){
				imageout_conv2[i][j][m]=Max(imageout_conv2[i][j][m]+bias_conv2[i],0);

			}
		}
	}

}
void MaxPool_layer2(data_type imagein_maxpool2[Depth1][N4][N4],data_type imageout2[Depth1][N5][N5])
{
#pragma HLS ARRAY_PARTITION variable=imagein_maxpool2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout2 complete dim=1

	MaxPool_layer2_label2:for(int j=0;j<N5;j++)
	{
		for(int k=0;k<N5;k++)
		{
			MaxPool_layer2_label1:for(int i=0;i<Depth1;i++)
			{
#pragma HLS UNROLL
				imageout2[i][j][k]=Max( Max( imagein_maxpool2[i][2*j][2*k], imagein_maxpool2[i][2*j][2*k+1]) , Max( imagein_maxpool2[i][2*j+1][2*k], imagein_maxpool2[i][2*j+1][2*k+1]) );//find the max of the pool
			}
		}
	}

}

void Convolution_layer3(data_type imagein_conv3[Depth1][N5][N5],data_type imageout_conv3[Depth2])
{

#pragma HLS ARRAY_PARTITION variable=imagein_conv3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Wconv3 cyclic factor=16 dim=1
	int i,j,m,n;
	data_type W3[Depth1][Depth2][M][M];
#pragma HLS ARRAY_PARTITION variable=W3 complete dim=1

			Convolution_layer3_label2:for(m=0;m<N5;m++){
				Convolution_layer3_label3:for(n=0;n<N5;n++){
					Convolution_layer3_label1:for(j=0;j<Depth2;j++){
#pragma HLS PIPELINE
					Convolution_layer3_label0:for(i=0;i<Depth1;i++){
									imageout_conv3[j]+=imagein_conv3[i][m][n]*Wconv3[i+j*16+m*9600+n*16*120];

				}
			}

		}
	}

	for(j=0;j<Depth2;j++){
		imageout_conv3[j]=Max(imageout_conv3[j]+bias_conv3[j],0);
	}
}

void fullconnected1(data_type imagein1[insize1],data_type imageout1[outsize1])
{
#pragma HLS ARRAY_PARTITION variable=imageout1 cyclic factor=7 dim=1
#pragma HLS ARRAY_PARTITION variable=W_fc1 cyclic factor=7 dim=1

	for(int j=0;j<outsize1;j++){
		imageout1[j]=bias1[j];
	}

	fullconnected1_label1:for(int i=0;i<insize1;i++){
	fullconnected1_label0:for(int j=0;j<outsize1;j++)
		{
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=7
			imageout1[j] += (imagein1[i] * W_fc1[i*84+j]);
		}

	}
	for(int j=0;j<outsize1;j++){
		imageout1[j] =Max(imageout1[j],0);
	}
}




void fullconnected2(data_type imagein2[insize2],data_type imageout2[outsize2])
{
#pragma HLS ARRAY_PARTITION variable=imagein2 cyclic factor=7 dim=1
#pragma HLS ARRAY_PARTITION variable=imageout2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W_fc2 cyclic factor=10 dim=1

	for(int j=0;j<outsize2;j++){
		imageout2[j]=bias2[j];
	}
fullconnected2_label1:for(int i=0;i<insize2;i++){
	fullconnected2_label0:for(int j=0;j<outsize2;j++)
	{

#pragma HLS UNROLL
	imageout2[j] += imagein2[i] * W_fc2[i*outsize2+j];
		}
	}
}
int8 cnn(data_type imagein[1024])
{
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both depth=1024 port=imagein
	//float adderin[784]={0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.14117648,0.6509804,0.9921569,0.9921569,0.9921569,0.9921569,0.7568628,0.21568629,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.14117648,0.32941177,0.9490197,0.93725497,0.65882355,0.62352943,0.24705884,0.24705884,0.41960788,0.16470589,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.16862746,0.7372549,0.96470594,0.57254905,0.06666667,0 ,0 ,0 ,0 ,0.03529412,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.14901961,0.67058825,0.9686275,0.4666667,0.05490196,0 ,0 ,0 ,0 ,0.38431376,0.8000001,0.27058825,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.75294125,0.98823535,0.73333335,0 ,0 ,0 ,0 ,0 ,0.2784314,0.8941177,0.9921569,0.3137255,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.14117648,1 ,0.8000001,0.03529412,0 ,0 ,0 ,0 ,0.08627451,0.9921569,0.9921569,0.7607844,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.82745105,0.93725497,0.26666668,0 ,0 ,0 ,0 ,0 ,0.39607847,0.98823535,0.98823535,0.2784314,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.16078432,0.909804,0.45098042,0 ,0 ,0 ,0 ,0.01568628,0.53333336,0.9490197,0.98823535,0.7803922,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.6627451,0.98823535,0.3137255,0 ,0 ,0 ,0 ,0.44705886,0.98823535,0.98823535,0.98823535,0.3019608,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.07058824,0.93725497,0.98823535,0 ,0 ,0 ,0 ,0.69411767,0.92549026,0.98823535,0.98823535,0.98823535,0.16470589,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.07058824,0.93725497,0.9921569,0.07058824,0 ,0.18823531,0.854902,0.9921569,0.9294118,0.9921569,0.9803922,0.61960787,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.52156866,0.98823535,0.85098046,0.74509805,0.9686275,0.9058824,0.3137255,0.40784317,0.98823535,0.9058824,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.1254902,0.83921576,0.909804,0.9058824,0.49411768,0.0627451,0 ,0.5803922,0.98823535,0.69803923,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.21176472,0.854902,0.9058824,0.20000002,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.82745105,0.9921569,0.65882355,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.14901961,0.92549026,0.90196085,0.10588236,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.2509804,0.98823535,0.82745105,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.45882356,0.98823535,0.79215693,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.6627451,0.98823535,0.4156863,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0.49411768,0.98823535,0.24313727,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 };
	data_type imagein_conv1[32][32];
	data_type imageout_conv1[Depth][N2][N2];
#pragma HLS ARRAY_PARTITION variable=imagein_conv1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv1 complete dim=2

	data_type imagein_conv2[Depth][N3][N3];
	data_type imageout_conv2[Depth1][N4][N4];
#pragma HLS ARRAY_PARTITION variable=imagein_conv2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=imageout_conv2 complete dim=1

	data_type imagein_conv3[Depth1][N5][N5];
	data_type imageout_conv3[Depth2];
#pragma HLS ARRAY_PARTITION variable=imagein_conv3 complete dim=1

	data_type imagein2[insize2];
	data_type imageout2[outsize2];
#pragma HLS ARRAY_PARTITION variable=imagein2 cyclic factor=7 dim=1
#pragma HLS ARRAY_PARTITION variable=imageout2 complete dim=1

	for(int j=0;j<32;j++){
		for(int m=0;m<32;m++){
			imagein_conv1[j][m]=imagein[j*32+m];
		}
	}

	Convolution_layer1(imagein_conv1, imageout_conv1);

	MaxPool_layer1(imageout_conv1,imagein_conv2);

	Convolution_layer2(imagein_conv2,imageout_conv2);

	MaxPool_layer2(imageout_conv2,imagein_conv3);

	Convolution_layer3(imagein_conv3,imageout_conv3);

	fullconnected1(imageout_conv3,imagein2);

	fullconnected2(imagein2,imageout2);
	data_type maxer=0;
	int8 no=0;
	for(int i=0;i<10;i++){
		if(maxer<imageout2[i]){
			no=i;
		}
		maxer=Max(maxer,imageout2[i]);

	}
	return no;


}


