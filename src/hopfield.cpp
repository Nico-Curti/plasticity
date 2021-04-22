#include <hopfield.h>

Hopfield :: Hopfield (const int & outputs, const int & batch_size, update_args optimizer,
                      weights_initialization weights_init,
                      int epochs_for_convergency, float convergency_atol,
                      float delta, float p, int k
                      ) : BasePlasticity (outputs, batch_size, transfer_t :: linear, optimizer, weights_init, epochs_for_convergency, convergency_atol),
                          k (k), delta (delta), p (p)
{
  this->check_params();
}


Hopfield :: Hopfield (const Hopfield & b) : BasePlasticity (b), k (b.k), delta (b.delta), p (b.p)
{
}

Hopfield & Hopfield :: operator = (const Hopfield & b)
{
  BasePlasticity :: operator = (b);

  this->k = b.k;
  this->delta = b.delta;
  this->p = b.p;

  return *this;
}


void Hopfield :: check_params ()
{
  // The value of the K variable must be positive and greater than 2
  if ( this->k < 2 )
  {
    std :: cerr << "k must be an integer bigger or equal than 2" << std :: endl;
    throw ERROR_K_POSITIVE;
  }
}


Eigen :: MatrixXf Hopfield :: weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output)
{
  // compute the columns re-order for the Krotov approximation

  // init the interaction matrix with a zero matrix
  Eigen :: MatrixXf yl = Eigen :: MatrixXf :: Zero(this->outputs, this->batch);
  int col_index = 0;

  // sort the output columns
  for (auto && col : output.colwise())
  {
    Eigen :: VectorXi order = Eigen :: VectorXi :: LinSpaced(this->outputs, 0, this->outputs);
    std :: sort(order.data(), order.data() + this->outputs,
                [&](const int & xi, const int & xj)
                {
                  return col[xi] < col[xj];
                });
    const int row_index = order[this->outputs - 1];
    yl(row_index, col_index) = 1.f;

    const int decrease_index = order[this->outputs - this->k];
    yl(decrease_index, col_index) = -this->delta;

    ++ col_index;
  }

  this->theta = yl.cwiseProduct(output).rowwise().sum();

  // compute the weights updates using the Hopfield formulation
  Eigen :: MatrixXf weights_update = (yl * X).array() - this->weights.array().colwise() * this->theta.array();

  // normalize the weights update by the maximum value
  // to avoid numerical instabilities
  const float max_abs_val = 1.f / weights_update.cwiseAbs().maxCoeff();
  weights_update = weights_update.array() * (-max_abs_val); // Add the minus for compatibility with optimization algorithms

  return weights_update;
}


void Hopfield :: normalize_weights ()
{
  // apply the Lebesgue norm to the weights matrix
  if ( this->p != 2.f )
    this->weights = this->weights.unaryExpr([&](const float & w) -> float {return std :: copysign(std :: pow(std :: fabs(w), this->p - 1.f), w);});
}


Eigen :: MatrixXf Hopfield :: _predict (const Eigen :: MatrixXf & data)
{
  // Compute the output as W @ X
  Eigen :: MatrixXf output = this->weights * data.transpose();

  return output;
}
