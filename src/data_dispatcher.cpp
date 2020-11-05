#include <data_dispatcher.h>
#include <base.h>

data_dispatcher :: data_dispatcher (float * buffer, const int & batch_size,
                                    const int & N_rows, const int & N_cols,
                                    bool shuffle) :
                                    batch (nullptr), indices (nullptr),
                                    num_batches (N_rows / batch_size), batch_dimension (batch_size * N_cols)
{

  if ( num_batches <= 0 )
  {
    std :: cerr << "Invalid batch size. The batch size must be less or equal to the number of samples." << std :: endl;
    std :: exit (ERROR_DISPATCH);
  }

  this->batch = new float * [this->num_batches];

  this->indices = std :: make_unique < int[] >(this->num_batches);
  std :: iota(this->indices.get(), this->indices.get() + this->num_batches, 0);

  if ( shuffle )
    std :: shuffle(this->indices.get(), this->indices.get() + this->num_batches, BasePlasticity :: engine);

  for (int i = 0; i < this->num_batches; ++i)
  {
    const int idx = this->indices[i];
    this->batch[i] = buffer + idx * N_cols * batch_size;
  }
}

data_dispatcher :: ~data_dispatcher ()
{
  delete [] this->batch;
  this->indices.reset();
}

float * data_dispatcher :: get_batch (const int & idx)
{
  if ( idx >= this->num_batches )
  {
    std :: cerr << "Index out of range. The current data dispatcher has " << this->num_batches << " batches. Given: " << idx << std :: endl;
    std :: exit(ERROR_DISPATCH);
  }

  return this->batch[idx];
}

void data_dispatcher :: randomize ()
{
  std :: shuffle(this->indices.get(), this->indices.get() + this->num_batches, BasePlasticity :: engine);

  for (int i = 0; i < this->num_batches; ++i)
  {
    const int idx = this->indices[i];
    std :: swap(this->batch[i], this->batch[idx]);
  }
}
