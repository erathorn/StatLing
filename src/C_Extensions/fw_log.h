#ifndef FW_LOG_H_H
#define FW_LOG_H_H


double forward_log(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size, int dimension,
                   double correction_parameter);
double random_score_c(char s1[], char s2[], long m, long n, double *gx_p);
double viterbi_log(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size, char *alignment);
double random_score_c_single(char s1[], long m, double *gx_p);
double random_score_c_single_tkf(char s1[], long m, double *gx_p, double mu, double l);
double viterbi_log_al(char s1[], char s2[], long m, long n, double *em_p, double *gx_p, double *gy_p, double *trans_p, int alphabet_size);

#endif // !FW_LOG_H_H

