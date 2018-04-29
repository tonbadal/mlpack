#include "gaussian_process.hpp"
#include <mlpack/core/util/log.hpp>

using namespace mlpack;
using namespace mlpack::gp;
using namespace distribution;

GaussianProcess::GaussianProcess(const arma::mat& predictors,
                  const arma::rowvec& responses,
                  const double sigma_n):
    sigma_n(sigma_n),
{
  Train(predictors, responses);
}

void GaussianProcess::Train(arma::mat predictors,
                            arma::vec responses)
{
  const size_t nCols = predictors.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  kernel = arma::zeros(nCols, nCols);
  double k;

  for(size_t i = 0 ; i < nCols ; i++)
  {
    for(size_t j = i ; j < nCols ; j++)
    {
      k = Kernel(predictors.col(i), predictors.col(j));
      kernel(i, j) = k;
      kernel(j, i) = k;
    }
  }
}

GaussianDistribution GaussianProcess::Predict(const arma::vec& point)
{
  const size_t nCols = predictors.n_cols;

  arma::vec kp;
  for(size_t i = 0 ; i < nCols ; i++ )
  {
    kp = Kernel(point, predictors.col(i));
  }
  double kpp = Kernel(point, point);

  arma::vec mean = arma::trans(kp) * arma::pinv(kernel + sigma_n * arma::eye(nCols, nCols)) * responses;
  arma::mat cov = kpp - arma::trans(kp) * arma::pinv(kernel + sigma_n * arma::eye(nCols, nCols)) * kp;

  return GaussianDistribution(mean, cov);
}


double GaussianProcess::Kernel(const arma::vec& x_i,
                               const arma::vec& x_j)
{
  const double sigma_f = 5;
  const double l = 0.2;

  arma::rowvec d = x_i - x_j;
  double k = sigma_f * sigma_f * exp(-0.5 * arma::norm(d) / l * l);

  return k;
}
