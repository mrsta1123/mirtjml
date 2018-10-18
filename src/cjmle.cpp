#include <RcppArmadillo.h>
#include <omp.h>

//' @useDynLib mirtjml
//' @importFrom Rcpp evalCpp

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec prox_func_cpp(const arma::vec &y, double C){
  double y_norm2 = arma::accu(square(y));
  if(y_norm2 <= C*C){
    return y;
  }
  else{
    return sqrt(C*C / y_norm2) * y;
  }
}
arma::vec prox_func_theta_cpp(arma::vec y, double C){
  double y_norm2 = arma::accu(square(y)) - 1;
  if(y_norm2 <= C*C-1){
    return y;
  }
  else{
    y = sqrt((C*C-1) / y_norm2) * y;
    y(0) = 1;
    return y;
  }
}

double neg_loglik(const arma::mat &thetaA, const arma::mat &response, const arma::mat &nonmis_ind){
  int N = response.n_rows;
  int J = response.n_cols;
  double res = arma::accu( nonmis_ind % (thetaA % response - log(1+exp(thetaA))) );
  return -res / N / J;
}

double neg_loglik_i_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i, const arma::mat &A, const arma::vec &theta_i){
  arma::vec tmp = A * theta_i;
  return -arma::accu(nonmis_ind_i % (tmp % response_i - log(1 + exp(tmp))));
}
arma::vec grad_neg_loglik_thetai_cpp(const arma::vec response_i, const arma::vec nonmis_ind_i, const arma::mat A, const arma::vec theta_i){
  arma::vec tmp = response_i - 1 / (1 + exp(- A * theta_i));
  arma::mat tmp1 = -arma::diagmat(nonmis_ind_i % tmp) * A;
  return( tmp1.t() * arma::ones(response_i.n_rows) );
}

// [[Rcpp::plugins(openmp)]]
arma::mat Update_theta_cpp(const arma::mat &theta0, const arma::mat &response, const arma::mat &nonmis_ind, const arma::mat &A0, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    double step = 1;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t());
    h(0) = 0;
    theta1.col(i) = theta0.row(i).t() - step * h;
    theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1.col(i)) > neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t()) &&
          step > 1e-4){
      step *= 0.5;
      theta1.col(i) = theta0.row(i).t() - step * h;
      theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    }
  }
  return(theta1.t());
}

double neg_loglik_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j, const arma::vec &A_j, const arma::mat &theta){
  arma::vec tmp = theta * A_j;
  return -arma::accu(nonmis_ind_j % (tmp % response_j - log(1+exp(tmp))));
}
arma::vec grad_neg_loglik_A_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j, const arma::vec &A_j, const arma::mat &theta){
  arma::vec tmp = response_j - 1 / (1 + exp(-theta * A_j));
  arma::vec tmp1 = nonmis_ind_j % tmp;
  arma::vec res = theta.row(0).t() * tmp1(0);
  for(unsigned int i=1;i<theta.n_rows;++i){
    res += theta.row(i).t() * tmp1(i);
  }
  return( -res );
}
// [[Rcpp::plugins(openmp)]]
arma::mat Update_A_cpp(const arma::mat &A0, const arma::mat &response, const arma::mat &nonmis_ind, const arma::mat &theta1, double C){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = 1;
    arma::vec h = grad_neg_loglik_A_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1);
    A1.col(j) = A0.row(j).t() - step * h;
    A1.col(j) = prox_func_cpp(A1.col(j), C);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1.col(j), theta1) > neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1) &&
          step > 1e-4){
      step *= 0.5;
      A1.col(j) = A0.row(j).t() - step * h;
      A1.col(j) = prox_func_cpp(A1.col(j), C);
    }
  }
  return(A1.t());
}
//' @export
// [[Rcpp::export]]
Rcpp::List alter_mini_CJMLE_cpp(const arma::mat &response, const arma::mat &nonmis_ind, arma::mat theta0,
                                arma::mat A0, double C, double tol, bool print_proc){
  // int N = theta0.n_rows;
  // int K = theta0.n_cols;
  // int J = A0.n_rows;
  arma::mat theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, C);
  arma::mat A1 = Update_A_cpp(A0, response, nonmis_ind, theta1, C);
  double eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
  while(eps > tol){
    if(print_proc) Rprintf("\n eps: %f", eps);
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, C);
    A1 = Update_A_cpp(A0, response, nonmis_ind, theta1, C);
    eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("obj") = neg_loglik(theta1*A1.t(), response, nonmis_ind));
}

