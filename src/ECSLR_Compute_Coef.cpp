// // [[Rcpp::depends(RcppEigen)]]
// // [[Rcpp::plugins(openmp)]]

#include <omp.h>
#include <RcppEigen.h>
#include <cmath>

using namespace Rcpp;
// using namespace RcppEigen;


// [[Rcpp::export]]
Eigen::MatrixXd Matrix_Mult(const Eigen::Map<Eigen::MatrixXd> &A,
                 const Eigen::Map<Eigen::MatrixXd> &B,
                 int n_cores=4){
  Eigen::setNbThreads(n_cores);
  Eigen::MatrixXd C = A * B;
  return C;
}

// [[Rcpp::export]]
Eigen::MatrixXd Matrix_crossprod(const Eigen::Map<Eigen::MatrixXd> A,
                      const Eigen::Map<Eigen::MatrixXd> B,
                      int n_cores=4){
  Eigen::setNbThreads(n_cores);
  Eigen::MatrixXd C = A.transpose() * B;
  return C;
}

// [[Rcpp::export]]
Eigen::MatrixXd Matrix_tcrossprod(const Eigen::Map<Eigen::MatrixXd> A,
                                 const Eigen::Map<Eigen::MatrixXd> B,
                                 int n_cores=4){
  Eigen::setNbThreads(n_cores);
  Eigen::MatrixXd C = A * B.transpose();
  return C;
}

// //' @export
// [[Rcpp::export]]
Eigen::VectorXd Soft(Eigen::Map<Eigen::VectorXd> z, Eigen::Map<Eigen::VectorXd> gamma) {
  return z.array().sign() * (z.array().abs() - gamma.array()).cwiseMax(0);
}


//' Sigmoid Function (Element-wise)
//'
//' Applies the sigmoid function element-wise to a numeric vector.
//' That is, it computes \eqn{1 / (1 + exp(-z_i))} for each element \eqn{z_i} of \code{z}.
//'
//' @param z A numeric vector.
//'
//' @return A numeric vector with the sigmoid of each element.
//'
//' @examples
//' sigmoid(c(-2, 0, 2))
//'
//' @export
// [[Rcpp::export]]
Eigen::VectorXd sigmoid(Eigen::Map<Eigen::VectorXd> z) {
  return static_cast<double>(1.0) / (1.0 + (-z).array().exp()); //must use 1.0: for double type calculation.
}

// [[Rcpp::export]]
double Compute_penalty_sparsity(int m, Eigen::Map<Eigen::MatrixXd> betas, double alpha_s, double lambda_sparsity) {
  return lambda_sparsity * (alpha_s * betas.col(m).array().abs().sum() + (1.0 - alpha_s) * 0.5 * betas.col(m).array().square().sum());
}

// [[Rcpp::export]]
double Compute_penalty_diversity(int m, Eigen::Map<Eigen::MatrixXd> betas, double lambda_diversity) {
  return lambda_diversity * 0.5 * (betas.col(m).array().abs() * (betas.array().abs().rowwise().sum() - betas.col(m).array().abs())).sum();
}

//' Compute Average Expected Cost (AEC)
//'
//' This function computes the Average Expected Cost (AEC) for binary classification,
//' given predicted probabilities, true labels, and a cost matrix.
//'
//' The AEC is calculated using the formula:
//' \deqn{
//' \text{AEC} = \frac{1}{n} \sum_{i=1}^n \Big[
//' y_i \cdot \big( s_i \cdot C_{TP,i} + (1 - s_i) \cdot C_{FN,i} \big) +
//' (1 - y_i) \cdot \big( s_i \cdot C_{FP,i} + (1 - s_i) \cdot C_{TN,i} \big)
//' \Big]
//' }
//' where \eqn{s_i} is the predicted score for instance \eqn{i}, and
//' \eqn{C_{TP}, C_{FP}, C_{TN}, C_{FN}} are costs of TP, FP, TN, FN, respectively.
//'
//' @param scores A numeric vector of predicted probabilities (values in [0, 1]).
//' @param y A binary vector of true labels (0 or 1).
//' @param cost_matrix A numeric matrix with 4 columns, where each row contains the costs
//' for TN, FP, FN, TP in this order.
//'
//' @return A scalar value representing the average expected cost.
//'
//' @examples
//' scores <- c(0.1, 0.7, 0.8)
//' y <- c(0, 1, 1)
//' cost_matrix <- matrix(c(0, 1, 3, 2,
//'                          0, 1, 6, 2,
//'                          0, 1, 9, 2), ncol = 4, byrow = TRUE)
//' Compute_AEC(scores, y, cost_matrix)
//'
//' @export
// [[Rcpp::export]]
double Compute_AEC(Eigen::Map<Eigen::VectorXd> scores,
                   Eigen::Map<Eigen::VectorXd> y,
                   Eigen::Map<Eigen::MatrixXd> cost_matrix) {
  return (y.array() * (scores.array() * cost_matrix.col(3).array() + (1 - scores.array()) * cost_matrix.col(2).array()) +
          (1 - y.array()) * (scores.array() * cost_matrix.col(1).array() + (1 - scores.array()) * cost_matrix.col(0).array())).mean();
}


// [[Rcpp::export]]
double Compute_objective(const int& m,
                         const Eigen::Map<Eigen::MatrixXd> x_std,
                         const Eigen::Map<Eigen::VectorXd> y,
                         const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                         const Eigen::Map<Eigen::MatrixXd> betas,
                         const Eigen::Map<Eigen::RowVectorXd> intercept,
                         const double& alpha_s,
                         const double& lambda_sparsity,
                         const double& lambda_diversity) {
  Eigen::VectorXd z = (x_std * betas.col(m)).array() + intercept[m];
  Eigen::Map<Eigen::VectorXd> mapped_z(z.data(), z.size());

  Eigen::VectorXd scores = sigmoid(mapped_z);
  Eigen::Map<Eigen::VectorXd> mapped_scores(scores.data(), scores.size());

  return (Compute_AEC(mapped_scores, y, cost_matrix)+
          Compute_penalty_sparsity(m, betas, alpha_s, lambda_sparsity)+
          Compute_penalty_diversity(m, betas, lambda_diversity));
}

// [[Rcpp::export]]
double Compute_objective_ensemble(const Eigen::Map<Eigen::MatrixXd> x_std,
                                  const Eigen::Map<Eigen::VectorXd> y,
                                  const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                                  const int& M,
                                  const Eigen::Map<Eigen::MatrixXd> betas,
                                  const Eigen::Map<Eigen::RowVectorXd> intercept,
                                  const double& alpha_s,
                                  const double& lambda_sparsity,
                                  const double& lambda_diversity) {
  Eigen::VectorXd obj_G = Eigen::VectorXd::Zero(M);
  #pragma omp parallel for
  for (int m = 0; m < M; ++m) {
    Eigen::VectorXd z = (x_std * betas.col(m)).array() + intercept[m]; // Calculate in parallel
    Eigen::Map<Eigen::VectorXd> mapped_z(z.data(), z.size());

    Eigen::VectorXd scores = sigmoid(mapped_z);
    Eigen::Map<Eigen::VectorXd> mapped_scores(scores.data(), scores.size());

    obj_G[m] =  Compute_AEC(mapped_scores, y, cost_matrix)+
      Compute_penalty_sparsity(m, betas, alpha_s, lambda_sparsity)+
      Compute_penalty_diversity(m, betas, lambda_diversity)/2; //due to the formula of Penalty for diversity.
  }
  return obj_G.sum();
}

// //' @export
// [[Rcpp::export]]
List ECSLR_Compute_Coef(const Eigen::Map<Eigen::MatrixXd> x,
                        const Eigen::Map<Eigen::VectorXd> y,
                        const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                        const int& M,
                        const bool& include_intercept,
                        const double& alpha_s,
                        const double& lambda_sparsity,
                        const double& lambda_diversity,
                        const double& tolerance,
                        const int& max_iter,
                        const std::string& model,
                        const Rcpp::Nullable<Rcpp::List> x0 = R_NilValue //,const int& n_threads
) {
  // Eigen::setNbThreads(n_threads);

  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd obj = Eigen::VectorXd::Constant(max_iter, -1);
  Eigen::VectorXd obj_G = Eigen::VectorXd::Zero(M);
  Eigen::VectorXd diff_com = Eigen::VectorXd::Constant(max_iter, -1);
  double obj_final=10000;
  // int cccount = 0;

  const double CCSA_RHOMIN = 1e-5;
  const double OBJ_TOLERANCE_REL = 1e-5;
  const double BETAS_TOLERANCE_ABS = tolerance; // Tolerance

  const int maxK = 100;
  const int maxL = 100;

  Eigen::MatrixXd betas(p,M);
  Eigen::RowVectorXd intercept(M);

  // Rcpp::Rcout << x0.size() << std::endl;
  if (x0.isNotNull()) {
    List x0_(x0); //transform to List object
    betas = x0_["betas"];
    intercept = x0_["intercept"];
  }else{
    betas = Eigen::MatrixXd::Zero(p, M);
    intercept = Eigen::RowVectorXd::Zero(1, M);
  }

  // Rcout <<"OBJ_TOLERANCE_REL="<<OBJ_TOLERANCE_REL<<", intercept=" << intercept.head(M) << std::endl;

  const Eigen::RowVectorXd mu_x = x.colwise().mean();
  const Eigen::RowVectorXd sd_x = ((x.rowwise() - mu_x).array().square().colwise().sum()/(n-1)).array().sqrt();
  Eigen::MatrixXd x_std = (x.rowwise() - mu_x).array().rowwise() / sd_x.array();
  x_std = x_std.array().isNaN().select(0.0, x_std.array());

  // initialize the parameter.
  Eigen::MatrixXd new_betas = betas;
  Eigen::RowVectorXd new_intercept = intercept;

  // define some map object.
  Eigen::Map<Eigen::MatrixXd> x_std_map(x_std.data(), x_std.rows(), x_std.cols());
  Eigen::Map<Eigen::MatrixXd> betas_map(betas.data(), betas.rows(), betas.cols());
  Eigen::Map<Eigen::RowVectorXd> intercept_map(intercept.data(), intercept.size());
  Eigen::Map<Eigen::MatrixXd> new_betas_map(new_betas.data(), new_betas.rows(), new_betas.cols());
  Eigen::Map<Eigen::RowVectorXd> new_intercept_map(new_intercept.data(), new_intercept.size());

  Eigen::MatrixXd betas_prev = betas;
  Eigen::RowVectorXd intercept_prev = intercept;

  Eigen::MatrixXd betas_prevprev = betas;
  Eigen::RowVectorXd intercept_prevprev = intercept;

  const Eigen::VectorXd b = y.array() * (cost_matrix.col(3) - cost_matrix.col(2)).array() +
    (1 - y.array()) * (cost_matrix.col(1) - cost_matrix.col(0)).array();

  const double denom_penalty = (1 - alpha_s) * lambda_sparsity;

  Eigen::RowVectorXd rho = Eigen::RowVectorXd::Ones(M);

  Eigen::VectorXd betas_mean(p);

  Eigen::RowVectorXd sigm0 = Eigen::RowVectorXd::Ones(M);
  Eigen::MatrixXd sigm = Eigen::MatrixXd::Ones(p, M);

  double obj_full=10000;
  double obj_full_new=obj_full;
  double dintercept2=0;
  double gam0=0;
  Eigen::VectorXd dbetas2(p);
  Eigen::VectorXd gam(p);

  Eigen::VectorXd z(n);
  Eigen::VectorXd expected_val_m(n);

  Eigen::VectorXd d(n);

  double aec_m = 0;

  Eigen::VectorXd new_z(n);
  Eigen::VectorXd new_expect_val(n);

  // Eigen::Map<Eigen::VectorXd> z_map(z.data(), z.size());
  // Eigen::Map<Eigen::VectorXd> expected_val_m_map(expected_val_m.data(), expected_val_m.size());
  // Eigen::Map<Eigen::VectorXd> new_z_map(new_z.data(), new_z.size());
  // Eigen::Map<Eigen::VectorXd> new_expect_val_map(new_expect_val.data(), new_expect_val.size());

  Eigen::Map<Eigen::VectorXd> z_map(z.data(), n);
  Eigen::Map<Eigen::VectorXd> expected_val_m_map(expected_val_m.data(), n);
  Eigen::Map<Eigen::VectorXd> new_z_map(new_z.data(), n);
  Eigen::Map<Eigen::VectorXd> new_expect_val_map(new_expect_val.data(), n);


  Eigen::VectorXd soft_penalty(p);
  Eigen::Map<Eigen::VectorXd> soft_penalty_map(soft_penalty.data(), soft_penalty.size());
  Eigen::VectorXd um(p);
  Eigen::Map<Eigen::VectorXd> um_map(um.data(), um.size());

  Eigen::VectorXd x_std_d(p);

  double grad_m=0;
  double w=0;
  double quad_m=0;
  double approxi_value=0;
  double actual_value=0;
  double delta=0;

  double max_diff=10000;
  double obj_rel_diff=10000;

  int r=0;

  // outer iteration
  for (r = 0; r < max_iter; ++r) {
    // Rcout << "r=" << r << std::endl;
    new_betas = betas;
    new_intercept = intercept;

    betas_mean = betas.rowwise().mean(); //apply(betas,1,mean)
    // MatrixXd betas.r = betas
    // MatrixXd intercept.r = intercept

    sigm0 = Eigen::RowVectorXd::Ones(M);
    sigm = Eigen::MatrixXd::Ones(p, M);

    for (int m = 0; m < M; ++m) {

      obj_full_new = Compute_objective(m, x_std_map,
                                       y, cost_matrix,
                                       betas_map, intercept_map,
                                       alpha_s,lambda_sparsity,lambda_diversity);

      for (int k = 1; k <= maxK; ++k) {
        // Rcout << "m:" << m << ", k:" << k << std::endl;

        obj_full = obj_full_new;

        if(k==1){
          rho[m] = 1.0;
        }else{
          rho[m] = std::max(0.1*rho[m],CCSA_RHOMIN);
        }
        // Rcout << "rho[m]=" << rho[m] << std::endl;

        if(k >=3){
          dintercept2 = (intercept[m] - intercept_prev[m]) * (intercept_prev[m] - intercept_prevprev[m]);
          gam0 = (dintercept2 < 0) ? 0.7 : ((dintercept2 > 0) ? 1.2 : 1.0);
          sigm0[m] = gam0 * sigm0[m];
          dbetas2 = (betas.col(m) - betas_prev.col(m)).array() * (betas_prev.col(m) - betas_prevprev.col(m)).array();
          // dbetas2 << -1, 0, 1;
          gam = dbetas2.array().unaryExpr([](double val) {
            return (val < 0) ? 0.7 : ((val > 0) ? 1.2 : 1.0);});
          sigm.col(m) = gam.array() * sigm.col(m).array();
        }

        z << (x_std * betas.col(m)).array() + intercept[m];
        // Eigen::Map<Eigen::VectorXd> z_map(z.data(), z.size());
        expected_val_m << sigmoid(z_map);
        // Eigen::Map<Eigen::VectorXd> expected_val_m_map(expected_val_m.data(), expected_val_m.size());
        aec_m = Compute_AEC(expected_val_m_map, y, cost_matrix);

        d = b.array() * expected_val_m.array() * (1-expected_val_m.array());


        soft_penalty << lambda_sparsity * alpha_s + lambda_diversity * 0.5 * (betas.cwiseAbs().rowwise().sum() - betas.col(m).cwiseAbs()).array();
        x_std_d = (static_cast<double>(1.0) / n) * (x_std.transpose() * d);

        // Rcout << "x_std_d: \n"<<  (x_std_d).head(10).transpose() << std::endl;
        // Rcout << "1/n x_std_d: \n"<<  (1/n * x_std_d).head(10).transpose() << std::endl;


        for (int l = 1; l <= maxL; ++l) {

          // cccount++;

          new_intercept[m] = include_intercept ? (intercept[m] - std::pow(sigm0[m],2)/rho[m] * d.mean()):(mu_x * (betas.col(m).array()/sd_x.transpose().array()).matrix());

          um << rho[m] * (betas.col(m).array() / sigm.col(m).array().square()) - x_std_d.array();
          new_betas.col(m) = (Soft(um_map, soft_penalty_map)).array() / (rho[m] / sigm.col(m).array().square() + denom_penalty);

          grad_m = (x_std_d.transpose() * (new_betas.col(m) - betas.col(m))).value() + d.mean() * (new_intercept[m] - intercept[m]);

          w = 0.5 * ( std::pow((new_intercept[m] - intercept[m])/sigm0[m],2)
                        + ((new_betas.col(m) - betas.col(m)).array() / sigm.col(m).array()).square().sum());
          quad_m = rho[m] * w;
          approxi_value = aec_m + grad_m + quad_m;

          new_z << (x_std * new_betas.col(m)).array() + new_intercept[m];
          // use "<<" instead of "=" so that the address of new_expect_val dose not change and the corresponding map can work.
          new_expect_val << sigmoid(new_z_map);

          actual_value = Compute_AEC(new_expect_val_map, y, cost_matrix);
          // Rcout << "l="<<  l <<  ", rho="<<  rho[m] <<  ", approxi_value=" << approxi_value << ", actual_value=" << actual_value << std::endl;

          // if approxi_value is upper bound, a better coefficients have been found
          if(approxi_value >= actual_value){
            break;
          }else{ // if approxi_value is not upper bound, change rho
            // update rho for iteration l+1
            delta = (actual_value-approxi_value)/w;
            rho[m] = std::min(10*rho[m],1.1*(rho[m] + delta));

            if(l==maxL){
              // When reaching the maxL, if no suitable solution is found, keep the original solution; Generally it is not used, most l will find a suitable solution within five cycles.
              // return the updated beta to its original value
              new_intercept[m] = intercept[m];
              new_betas.col(m) = betas.col(m);
              break;
              // approxi_value = aec_m
            }
          }
        } // end of inner iteration for l

        max_diff = std::max( (new_betas.col(m)-betas.col(m)).cwiseAbs().maxCoeff(), std::abs(new_intercept[m]-intercept[m]));

        // approxi_full = aec_m + Compute_penalty_sparsity(m, betas,alpha_s,lambda_sparsity) + Compute_penalty_diversity(m,betas,lambda_diversity)
        // approxi_full_new = approxi_value + Compute_penalty_sparsity(m, new_betas,alpha_s,lambda_sparsity) + Compute_penalty_diversity(m,new_betas,lambda_diversity)
        // cat("approxi_full",approxi_full,", approxi_full_new:",approxi_full_new,"\n")

        obj_full_new = Compute_objective(m, x_std_map, y, cost_matrix,
                                         new_betas_map, new_intercept_map,
                                         alpha_s, lambda_sparsity, lambda_diversity);

        obj_rel_diff = (obj_full_new - obj_full)/obj_full;

        // Rcout << "m:" << m << ", k:" << k << ", objective on betas:" << obj_full << ", objective on newbetas:" << obj_full_new << std::endl;
        // Rcout << "m:" << m << ", k:" << k << ", max.betas.diff=" << max_diff << ", rel_diff=" << obj_rel_diff << std::endl;

        // didn't become better, use old solution, do not update betas and intercept
        if(obj_full_new > obj_full){

          new_betas.col(m) = betas.col(m);
          new_intercept[m] = intercept[m];
          //didn't become better, use old solution, do not update betas and intercept
          break;
        }

        // update betas and intercept m
        if(max_diff < BETAS_TOLERANCE_ABS || ( std::abs(obj_rel_diff) < OBJ_TOLERANCE_REL ) ){
          betas.col(m) = new_betas.col(m);
          intercept[m] = new_intercept[m];
          break;
        }else{
          if(k >=2){
            betas_prevprev.col(m) = betas_prev.col(m);
            intercept_prevprev[m] = intercept_prev[m];
          }

          betas_prev.col(m) = betas.col(m);
          intercept_prev[m] = intercept[m];

          betas.col(m) = new_betas.col(m);
          intercept[m] = new_intercept[m];
        }

      } // end of outer iteration for k

    } // end of m

    // betas has been update during calculation
    // #pragma omp parallel for
    // for (int m = 0; m < M; ++m) {
    //   obj_G[m] = Compute_objective(m, x_std_map, y, cost_matrix, betas_map, intercept_map, alpha_s, lambda_sparsity, lambda_diversity);
    // }
    // obj[r] = obj_G.sum();

    obj[r] = Compute_objective_ensemble(x_std_map, y, cost_matrix,
                                        M,
                                        betas_map, intercept_map,
                                        alpha_s, lambda_sparsity, lambda_diversity);

    // diff_com = max(abs(betas-betas.r), abs(intercept-intercept.r))
    // Rcout << "betas_mean=" << betas_mean << std::endl;
    // Rcout << "betas.rowwise().mean()=" << betas.rowwise().mean() << std::endl;

    diff_com[r] = (betas.rowwise().mean()-betas_mean).array().square().maxCoeff();
    // Rcout << "r=" << r << ", difference between betas and newbetas: " << diff_com[r] << ", obj=" << obj[r] << std::endl;

    if(diff_com[r] < tolerance){
      break;
    }
  }

  int num = std::min(r+1,max_iter);
  // Eigen::VectorXd obj = obj.head(num);
  obj_final = obj[num-1];
  // Eigen::VectorXd diff_com = diff_com.head(num);

  // Rcout << "obj=" << obj.head(num).transpose()  << std::endl;
  // Rcout << "diff_com=" << diff_com.head(num).transpose()  << std::endl;

  Eigen::MatrixXd betas_scaled = betas.array().colwise() / sd_x.transpose().array();
  betas_scaled = betas_scaled.unaryExpr([](double val) {
    return std::isnan(val) || std::isinf(val) ? 0.0 : val;
  });

  // Rcout << "count=" << cccount << "\n"<< std::endl;

  Eigen::RowVectorXd intercept_scaled = (include_intercept? 1: 0) * (intercept - mu_x * betas_scaled);

  Eigen::VectorXd betas_est = betas_scaled.rowwise().mean();
  double intercept_est = intercept_scaled.mean();

  return List::create(Named("betas_scaled") = betas_scaled,
                      Named("intercept_scaled") = intercept_scaled,
                      Named("betas_est") = betas_est,
                      Named("intercept_est") = intercept_est,
                      Named("lambda_sparsity") = lambda_sparsity,
                      Named("lambda_diversity") = lambda_diversity,
                      Named("alpha_s") = alpha_s,
                      Named("M") = M,
                      Named("include_intercept") = include_intercept,
                      Named("model") = model,
                      Named("objective") = obj_final,
                      Named("coef_x_std") = List::create(
                        Named("betas") = betas,
                        Named("intercept") = intercept
                      )
  );
}
