#include <bcm.h>

BCM :: BCM (const int & outputs, const int & batch_size,
            int activation, update_args optimizer, weights_initialization weights_init,
            int epochs_for_convergency, float convergency_atol,
            float decay, float memory_factor,
            float interaction_strength
            ) : BasePlasticity (outputs, batch_size, activation, optimizer, weights_init, epochs_for_convergency, convergency_atol, decay)
{
  this->init_interaction_matrix(interaction_strength);
  this->memory_factor = memory_factor;
}


BCM :: BCM (const BCM & b) : BasePlasticity (b)
{
  this->memory_factor = memory_factor;
}

BCM & BCM :: operator = (const BCM & b)
{
  BasePlasticity :: operator = (b);
  this->memory_factor = memory_factor;

  return *this;
}


void BCM :: init_interaction_matrix (const float & interaction_strength)
{
  if (interaction_strength != 0.f)
  {
    // create a temporary matrix with all the elements set as -interaction_strength
    Eigen :: MatrixXf temp = Eigen :: MatrixXf :: Constant(this->outputs, this->outputs, -interaction_strength);

    // // local interaction setting a symmetric matrix
    // Eigen :: MatrixXf symm = Eigen :: MatrixXf :: Zero(this->outputs, this->outputs);
    // for (int32_t i = 0; i < this->outputs; ++i)
    //   for (int32_t j = i + 1; j < this->outputs; ++j)
    //     symm(i, j) = (this->outputs + i - j) * (-interaction_strength);
    //
    // Eigen :: MatrixXf temp = symm + symm.transpose();

    // re-fill only the diagonal values with the identity
    temp.diagonal() = Eigen :: ArrayXf :: Ones(this->outputs);

    // invert the matrix
    this->interaction_matrix = temp.inverse();
  }

  else
    // without the lateral interactions the interaction_matrix is just the inverse of
    // the identity matrix, i.e the identity matrix!
    this->interaction_matrix = Eigen :: MatrixXf :: Identity(this->outputs, this->outputs);
}


Eigen :: MatrixXf BCM :: weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output)
{
  // evaluate the theta array as the average of the output rows
  Eigen :: VectorXf theta = output.array().square().rowwise().mean();

  // update the theta array with the moving average
  this->theta = this->memory_factor * this->theta + (1.f - this->memory_factor) * theta;

  // compute the phi array, i.e the Law and Cooper function
  // Step 1 : φ = y * (y - θ)
  // Step 2: φ = φ / θ
  Eigen :: MatrixXf phi = output.cwiseProduct(output.colwise() - theta);
  // NOTE: add an extra epsilon term in the denominator to avoid possible numerical issues
  phi = phi.array().colwise() / (theta.array() + BasePlasticity :: precision);

  // compute the weights update using Law and Cooper rule
  // dw/dt = φ * x
  Eigen :: MatrixXf weights_update = phi * X;

  // normalize the weights update according to the number of samples
  const float max_abs_val = 1.f / X.rows();
  weights_update = weights_update.array() * (-max_abs_val); // Add the minus for compatibility with optimization algorithms

  return weights_update;
}



Eigen :: MatrixXf BCM :: _predict (const Eigen :: MatrixXf & data)
{
  // Apply the interaction matrix to the weights and compute the output
  Eigen :: MatrixXf output = this->interaction_matrix * this->weights * data.transpose();

  // apply the activation function
  output = output.unaryExpr([&](const float & out) -> float {return this->activation(out);});

  return output;
}
