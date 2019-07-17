#include <stdlib.h>
#include <math.h>
#include "fw_log.h"
double exp(double);
double log(double);


struct alignment_mat
{
  double value;
  int prevX, prevY, prevZ;
};




double forward_log(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size, int dimension,
                    double correction_parameter) {
	double inf = INFINITY;
	double P = 0.0;
	double trellis[m+1][n+1][3];
    int offset=dimension*alphabet_size*alphabet_size;
    int offset_tr = dimension*16;
    double v, xv, yv;
	size_t i, j, x, y, k;

    double t00 = *(trans_p+offset_tr);
    double t01 = *(trans_p+offset_tr+1);
    double t02 = *(trans_p+offset_tr+2);
    double t0E = *(trans_p+offset_tr+3);
    double t10 = *(trans_p+offset_tr+4);
    double t11 = *(trans_p+offset_tr+5);
    double t12 = *(trans_p+offset_tr+6);
    double t1E = *(trans_p+offset_tr+7);
    double t20 = *(trans_p+offset_tr+8);
    double t21 = *(trans_p+offset_tr+9);
    double t22 = *(trans_p+offset_tr+10);
    double t2E = *(trans_p+offset_tr+11);
    double tS0 = *(trans_p+offset_tr+12);
    double tS1 = *(trans_p+offset_tr+13);
    double tS2 = *(trans_p+offset_tr+14);


    for (i=0; i< m+1; i++) {

        for (j=0; j< n+1; j++) {
            for (k =0; k < 3; k++){
                trellis[i][j][k] = -inf;
            };
        };
    };

    trellis[0][0][0] = tS0;
	trellis[0][0][1] = tS1;
	trellis[0][0][2] = tS2;

    x = (size_t) s1[0];
	trellis[1][0][1] = *(gx_p+x) + log(exp(trellis[0][0][0]+t01) + exp(trellis[0][0][1]+t11)+ exp(trellis[0][0][2]+t21));


	for (i = 2; i < m + 1; i++) {
	    x = (size_t) s1[i-1];
	    trellis[i][0][1] = *(gx_p+x) + trellis[i - 1][0][1] + t11;
	};


    y = (size_t) s2[0];
	trellis[0][1][2] = *(gy_p+y) + log(exp(trellis[0][0][0]+t02) + exp(trellis[0][0][2]+t22)+ exp(trellis[0][0][1]+t12));
	for (i = 2; i < n + 1; i++){
	    y = (size_t) s2[i-1];
		trellis[0][i][2] = *(gy_p+y) + trellis[0][i - 1][2] + t22;
	};


	for (i = 1; i < m+1; i++){
	    x = (size_t) s1[i - 1];
	    xv = *(gx_p+x);
		for (j = 1; j < n+1 ; j++){

		    y = (size_t) s2[j - 1];
		    yv = *(gy_p+y);

		    v = *(em_p+offset+x*alphabet_size+y);

			trellis[i][j][0] = v + log(exp(t00 + trellis[i - 1][j - 1][0]) + exp(t10+trellis[i - 1][j - 1][1]) + exp(t20 + trellis[i - 1][j - 1][2]));
			trellis[i][j][1] = xv + log(exp(trellis[i - 1][j][0] + t01) + exp(trellis[i - 1][j][1] + t11) + exp(trellis[i - 1][j][2] + t21));
			trellis[i][j][2] = yv + log(exp(trellis[i][j - 1][0] + t02) + exp(trellis[i][j - 1][2] + t22) + exp(trellis[i][j - 1][1] + t12));

		};
	};
	P = log(exp(t0E + trellis[m][n][0]) + exp(t1E + trellis[m][n][1]) + exp(t2E + trellis[m][n][2]));

	return P;
}



double random_score_c_single(char s1[], long m, double *gx_p){
    double a_ = 0;
    size_t i, x;

    for (i = 0; i < m; i++) {
	    x = (size_t) s1[i];
	    a_ += *(gx_p+x);
    };
    return a_;
}

double random_score_c_single_tkf(char s1[], long m, double *gx_p, double mu, double l){
    double a_ = 0;
    size_t i, x;
    if (mu != -1){
        a_ = log(1.0-(l/mu)) + m*log(l/mu);
        };
    a_ += random_score_c_single(s1, m, gx_p);

    return a_;
}

double random_score_c(char s1[], char s2[], long m, long n, double *gx_p){
    double a_ = 0;
    size_t i, x;

    for (i = 0; i < m; i++) {
	    x = (size_t) s1[i];
	    a_ += *(gx_p+x);
    };
    for (i = 0; i < n; i++) {
	    x = (size_t) s2[i];
	    a_ += *(gx_p+x);
    };
    return a_;
}


double viterbi_log_al(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size) {
	double inf = INFINITY;
	double P = 0.0;
	struct alignment_mat trellis[m+1][n+1][3];
    int tmpX, tmpY, tmpZ ,t_tmpX, t_tmpY, t_tmpZ;
    double alignment_length, v,m_s, x_s, y_s;
	size_t i, j, x, y, k;

    double t00 = *trans_p;
    double t01 = *(trans_p+1);
    double t02 = *(trans_p+2);
    double t0E = *(trans_p+3);
    double t10 = *(trans_p+4);
    double t11 = *(trans_p+5);
    double t12 = *(trans_p+6);
    double t1E = *(trans_p+7);
    double t20 = *(trans_p+8);
    double t21 = *(trans_p+9);
    double t22 = *(trans_p+10);
    double t2E = *(trans_p+11);
    double tS0 = *(trans_p+12);
    double tS1 = *(trans_p+13);
    double tS2 = *(trans_p+14);

    // Initialize Trellis
    //for (i=0; i< m+1; i++) {

    //    for (j=0; j< n+1; j++) {
    //        for (k =0; k < 3; k++){
    //            trellis[i][j][k].value = -inf;
    //            trellis[i][j][k].prevZ = 0;
    //            trellis[i][j][k].prevX = 0;
    //            trellis[i][j][k].prevY = 0;
    //        };
    //    };
    //};

    // set initial values
    trellis[0][0][0].value = tS0;
    trellis[0][0][0].prevZ = -1;
    trellis[0][0][0].prevX = -1;
    trellis[0][0][0].prevY = -1;

	trellis[0][0][1].value = tS1;
	trellis[0][0][1].prevZ = -1;
    trellis[0][0][1].prevX = -1;
    trellis[0][0][1].prevY = -1;

	trellis[0][0][2].value = tS2;
	trellis[0][0][2].prevZ = -1;
    trellis[0][0][2].prevX = -1;
    trellis[0][0][2].prevY = -1;

    for (i=1; i< m+1; i++) {
        trellis[i][0][0].value = -inf;
        trellis[i][0][0].prevX = i-1;
        trellis[i][0][0].prevY = 0;
        trellis[i][0][0].prevZ = 0;

        trellis[i][0][2].value = -inf;
        trellis[i][0][2].prevX = i-1;
        trellis[i][0][2].prevY = 0;
        trellis[i][0][2].prevZ = 2;
    };

    x = (size_t) s1[0];
    m_s = t01 + trellis[0][0][0].value;
    x_s = t11 + trellis[0][0][1].value;
    y_s = t21 + trellis[0][0][2].value;

    if (m_s >= x_s && m_s >= y_s){
	    trellis[1][0][1].value = *(gx_p+x) + m_s;
	    trellis[1][0][1].prevZ = 0;
	    }
	else if (y_s >= x_s){
	    trellis[1][0][1].value = *(gx_p+x) + y_s;
	    trellis[1][0][1].prevZ = 2;
	    }
	else{
	    trellis[1][0][1].value = *(gx_p+x) + x_s;
	    trellis[1][0][1].prevZ = 1;
	    };
    trellis[1][0][1].prevX = 0;
    trellis[1][0][1].prevY = 0;

	for (i = 2; i < m + 1; i++) {
	    x = (size_t) s1[i-1];
		trellis[i][0][1].value = *(gx_p+x) + t11 + trellis[i - 1][0][1].value;
		trellis[i][0][1].prevX = i-1;
		trellis[i][0][1].prevY = 0;
		trellis[i][0][1].prevZ = 1;
	};


    for (i=1; i< n+1; i++) {
        trellis[0][i][0].value = -inf;
        trellis[0][i][0].prevX = 0;
        trellis[0][i][0].prevY = i-1;
        trellis[0][i][0].prevZ = 0;


        trellis[0][i][1].value = -inf;
        trellis[0][i][1].prevX = 0;
        trellis[0][i][1].prevY = i-1;
        trellis[0][i][1].prevZ = 1;
    };
    m_s = t02 + trellis[0][0][0].value;
    x_s = t12 + trellis[0][0][1].value;
    y_s = t22 + trellis[0][0][2].value;

    y = (size_t) s2[0];

    if (m_s >= y_s && m_s >= x_s){
	    trellis[0][1][2].value = *(gy_p+y) + m_s;
	    trellis[0][1][2].prevZ = 0;
	    }
	else if (x_s >= y_s){
	    trellis[0][1][2].value = *(gy_p+y) + x_s;
	    trellis[0][1][2].prevZ = 1;
	    }
	else{
	    trellis[0][1][2].value = *(gy_p+y) + y_s;
	    trellis[0][1][2].prevZ = 2;
	    };
    trellis[0][1][2].prevX = 0;
    trellis[0][1][2].prevY = 0;
	for (i = 2; i < n + 1; i++)
	{
	    y = (size_t) s2[i-1];
		trellis[0][i][2].value = *(gy_p+y) + t22 + trellis[0][i - 1][2].value;
		trellis[0][i][2].prevX = 0;
        trellis[0][i][2].prevY = i-1;
        trellis[0][i][2].prevZ = 2;

	};

	for (i = 1; i < m + 1; i++)
	{
		for (j = 1; j < n + 1; j++)
		{
			x = (size_t) s1[i - 1];
			y = (size_t) s2[j - 1];

            v = *(em_p + x * alphabet_size + y);
            m_s = t00 + trellis[i - 1][j - 1][0].value;
            x_s = t10 + trellis[i - 1][j - 1][1].value;
            y_s = t20 + trellis[i - 1][j - 1][2].value;
            if (m_s >= x_s && m_s >= y_s) {
			    trellis[i][j][0].value = v + m_s;
			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 0;
			    }
			else if (x_s >= y_s){
			    trellis[i][j][0].value = v + x_s;

			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 1;
			    }
			else {
			    trellis[i][j][0].value = v + y_s;

			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 2;
			};
			m_s = t01 + trellis[i - 1][j][0].value;
            x_s = t11 + trellis[i - 1][j][1].value;
            y_s = t21 + trellis[i - 1][j][2].value;

            if (m_s >= x_s && m_s >= y_s){
                trellis[i][j][1].value = *(gx_p+x) + m_s;

			    trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 0;
                }
            else if (y_s >= x_s){
                trellis[i][j][1].value = *(gx_p+x) + y_s;

                trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 2;
            }
            else{
                trellis[i][j][1].value = *(gx_p+x) + x_s;

                trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 1;
                };
			m_s = t02 + trellis[i][j-1][0].value;
			x_s = t12 + trellis[i][j-1][1].value;
            y_s = t22 + trellis[i][j-1][2].value;

            if (m_s >= x_s && m_s >= y_s){
                trellis[i][j][2].value = *(gy_p+y) + m_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 0;
                }
            else if (x_s >= y_s){
                trellis[i][j][2].value = *(gy_p+y) + x_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 1;
                }
            else{
                trellis[i][j][2].value = *(gy_p+y) + y_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 2;
                };

		};
	};
	m_s = trellis[m][n][0].value;
	x_s = trellis[m][n][1].value;
	y_s = trellis[m][n][2].value;
	if (m_s >= x_s && m_s >= y_s) {
	    P = m_s+t0E;
	    tmpZ = 0;
	    }
	else if (x_s >= y_s){
	    P = t1E + x_s;
	    tmpZ = 1;
	    }
	else {
	    P = t2E + y_s;
	    tmpZ = 2;
	    };

	tmpX = m;
	tmpY = n;
	alignment_length = 0;
    while(tmpX > 0 || tmpY > 0){
        alignment_length += 1;
        t_tmpX = trellis[tmpX][tmpY][tmpZ].prevX;
        t_tmpY = trellis[tmpX][tmpY][tmpZ].prevY;
        t_tmpZ = trellis[tmpX][tmpY][tmpZ].prevZ;
        tmpX = t_tmpX;
        tmpY = t_tmpY;
        tmpZ = t_tmpZ;
    };

	return alignment_length;
}


double viterbi_log(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size,
                    char *alignment) {
	double inf = INFINITY;
	double P = 0.0;
	struct alignment_mat trellis[m+1][n+1][3];
    int tmpX, tmpY, tmpZ ,t_tmpX, t_tmpY, t_tmpZ, ch_s, ch_f;
    double alignment_length, v,m_s, x_s, y_s;
	size_t i, j, x, y, k;

    double t00 = *trans_p;
    double t01 = *(trans_p+1);
    double t02 = *(trans_p+2);
    double t0E = *(trans_p+3);
    double t10 = *(trans_p+4);
    double t11 = *(trans_p+5);
    double t12 = *(trans_p+6);
    double t1E = *(trans_p+7);
    double t20 = *(trans_p+8);
    double t21 = *(trans_p+9);
    double t22 = *(trans_p+10);
    double t2E = *(trans_p+11);
    double tS0 = *(trans_p+12);
    double tS1 = *(trans_p+13);
    double tS2 = *(trans_p+14);

    // Initialize Trellis
    //for (i=0; i< m+1; i++) {

    //    for (j=0; j< n+1; j++) {
    //        for (k =0; k < 3; k++){
    //            trellis[i][j][k].value = -inf;
    //            trellis[i][j][k].prevZ = 0;
    //            trellis[i][j][k].prevX = 0;
    //            trellis[i][j][k].prevY = 0;
    //        };
    //    };
    //};



    // set initial values
    trellis[0][0][0].value = tS0;
    trellis[0][0][0].prevZ = -1;
    trellis[0][0][0].prevX = -1;
    trellis[0][0][0].prevY = -1;

	trellis[0][0][1].value = tS1;
	trellis[0][0][1].prevZ = -1;
    trellis[0][0][1].prevX = -1;
    trellis[0][0][1].prevY = -1;

	trellis[0][0][2].value = tS2;
	trellis[0][0][2].prevZ = -1;
    trellis[0][0][2].prevX = -1;
    trellis[0][0][2].prevY = -1;

    for (i=1; i< m+1; i++) {
        trellis[i][0][0].value = -inf;
        trellis[i][0][0].prevX = i-1;
        trellis[i][0][0].prevY = 0;
        trellis[i][0][0].prevZ = 0;

        trellis[i][0][2].value = -inf;
        trellis[i][0][2].prevX = i-1;
        trellis[i][0][2].prevY = 0;
        trellis[i][0][2].prevZ = 2;
    };

    x = (size_t) s1[0];
    m_s = t01 + trellis[0][0][0].value;
    x_s = t11 + trellis[0][0][1].value;
    y_s = t21 + trellis[0][0][2].value;

    if (m_s > x_s && m_s > y_s){
	    trellis[1][0][1].value = *(gx_p+x) + m_s;
	    trellis[1][0][1].prevZ = 0;
	    }
	else if (y_s > x_s){
	    trellis[1][0][1].value = *(gx_p+x) + y_s;
	    trellis[1][0][1].prevZ = 2;
	    }
	else{
	    trellis[1][0][1].value = *(gx_p+x) + x_s;
	    trellis[1][0][1].prevZ = 1;
	    };
    trellis[1][0][1].prevX = 0;
    trellis[1][0][1].prevY = 0;

	for (i = 2; i < m + 1; i++) {
	    x = (size_t) s1[i-1];
		trellis[i][0][1].value = *(gx_p+x) + t11 + trellis[i - 1][0][1].value;
		trellis[i][0][1].prevX = i-1;
		trellis[i][0][1].prevY = 0;
		trellis[i][0][1].prevZ = 1;
	};


    for (i=1; i< n+1; i++) {
        trellis[0][i][0].value = -inf;
        trellis[0][i][0].prevX = 0;
        trellis[0][i][0].prevY = i-1;
        trellis[0][i][0].prevZ = 0;


        trellis[0][i][1].value = -inf;
        trellis[0][i][1].prevX = 0;
        trellis[0][i][1].prevY = i-1;
        trellis[0][i][1].prevZ = 1;
    };
    m_s = t02 + trellis[0][0][0].value;
    x_s = t12 + trellis[0][0][1].value;
    y_s = t22 + trellis[0][0][2].value;

    y = (size_t) s2[0];

    if (m_s > y_s && m_s > x_s){
	    trellis[0][1][2].value = *(gy_p+y) + m_s;
	    trellis[0][1][2].prevZ = 0;
	    }
	else if (x_s > y_s){
	    trellis[0][1][2].value = *(gy_p+y) + x_s;
	    trellis[0][1][2].prevZ = 1;
	    }
	else{
	    trellis[0][1][2].value = *(gy_p+y) + y_s;
	    trellis[0][1][2].prevZ = 2;
	    };
    trellis[0][1][2].prevX = 0;
    trellis[0][1][2].prevY = 0;
	for (i = 2; i < n + 1; i++)
	{
	    y = (size_t) s2[i-1];
		trellis[0][i][2].value = *(gy_p+y) + t22 + trellis[0][i - 1][2].value;
		trellis[0][i][2].prevX = 0;
        trellis[0][i][2].prevY = i-1;
        trellis[0][i][2].prevZ = 2;

	};

	for (i = 1; i < m + 1; i++)
	{
		for (j = 1; j < n + 1; j++)
		{
			x = (size_t) s1[i - 1];
			y = (size_t) s2[j - 1];

            v = *(em_p + x * alphabet_size + y);
            m_s = t00 + trellis[i - 1][j - 1][0].value;
            x_s = t10 + trellis[i - 1][j - 1][1].value;
            y_s = t20 + trellis[i - 1][j - 1][2].value;
            if (m_s > x_s && m_s > y_s) {
			    trellis[i][j][0].value = v + m_s;
			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 0;
			    }
			else if (x_s > y_s){
			    trellis[i][j][0].value = v + x_s;

			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 1;
			    }
			else {
			    trellis[i][j][0].value = v + y_s;

			    trellis[i][j][0].prevX = i-1;
                trellis[i][j][0].prevY = j-1;
                trellis[i][j][0].prevZ = 2;
			};
			m_s = t01 + trellis[i - 1][j][0].value;
            x_s = t11 + trellis[i - 1][j][1].value;
            y_s = t21 + trellis[i - 1][j][2].value;

            if (m_s > x_s && m_s > y_s){
                trellis[i][j][1].value = *(gx_p+x) + m_s;

			    trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 0;
                }
            else if (y_s > x_s){
                trellis[i][j][1].value = *(gx_p+x) + y_s;

                trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 2;
            }
            else{
                trellis[i][j][1].value = *(gx_p+x) + x_s;

                trellis[i][j][1].prevX = i-1;
                trellis[i][j][1].prevY = j;
                trellis[i][j][1].prevZ = 1;
                };
			m_s = t02 + trellis[i][j-1][0].value;
			x_s = t12 + trellis[i][j-1][1].value;
            y_s = t22 + trellis[i][j-1][2].value;

            if (m_s > x_s && m_s > y_s){
                trellis[i][j][2].value = *(gy_p+y) + m_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 0;
                }
            else if (x_s > y_s){
                trellis[i][j][2].value = *(gy_p+y) + x_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 1;
                }
            else{
                trellis[i][j][2].value = *(gy_p+y) + y_s;

                trellis[i][j][2].prevX = i;
                trellis[i][j][2].prevY = j-1;
                trellis[i][j][2].prevZ = 2;
                };

		};
	};
	m_s = trellis[m][n][0].value;
	x_s = trellis[m][n][1].value;
	y_s = trellis[m][n][2].value;
	if (m_s >= x_s && m_s >= y_s) {
	    P = m_s+t0E;
	    tmpZ = 0;
	    }
	else if (x_s >= y_s){
	    P = t1E + x_s;
	    tmpZ = 1;
	    }
	else {
	    P = t2E + y_s;
	    tmpZ = 2;
	    };

	tmpX = m;
	tmpY = n;
	alignment_length = 0;
	ch_f = 0;
	ch_s = 0;
    while(tmpX > 0 || tmpY > 0){
        alignment_length += 1;
        if (tmpZ == 2){
            alignment[ch_f] = -1;
            alignment[34+ch_s] = s2[tmpY-1];
        } else if (tmpZ == 1){
            alignment[ch_f] = s1[tmpX-1];
            alignment[34+ch_s] = -1;
        } else {
            alignment[ch_f] = s1[tmpX-1];
            alignment[34+ch_s] = s2[tmpY-1];
        };
        ch_f += 1;
        ch_s += 1;
        t_tmpX = trellis[tmpX][tmpY][tmpZ].prevX;
        t_tmpY = trellis[tmpX][tmpY][tmpZ].prevY;
        t_tmpZ = trellis[tmpX][tmpY][tmpZ].prevZ;
        tmpX = t_tmpX;
        tmpY = t_tmpY;
        tmpZ = t_tmpZ;
    };

	return P;
}