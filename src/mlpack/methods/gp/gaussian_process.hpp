/**
 * @file gaussian_process.hpp
 * @author Ton Badal
 *
 * Gaussian process.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_GP_GAUSSIAN_PROCESS_HPP
#define MLPACK_METHODS_GP_GAUSSIAN_PROCESS_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/dists/gaussian_distribution.hpp>

namespace mlpack{
namespace gp{

class GaussianProcess
{
 public:
  /**
   * Creates the model.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param l lengthscale, distance between turning points in the function.
   * @param sigma_f, marginal variance.
   */
  GaussianProcess(const arma::mat& predictors,
                  const arma::rowvec& responses,
                  const double sigma_n = 0);

  void Train(arma::mat predictors,
             arma::vec responses);

  distribution::GaussianDistribution Predict(const arma::vec& point);


 private:
  arma::mat predictors;
  arma::vec responses;
  double sigma_n;
  arma::mat kernel;

  double Kernel(const arma::vec& x_i,
                const arma::vec& x_j);

};

} // namespace gp
} // namespace mlpack


#endif // MLPACK_METHODS_GP_GAUSSIAN_PROCESS_HPP
