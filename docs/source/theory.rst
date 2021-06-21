Theory
======

The BCM model describes the synaptic plasticity via a dynamic adaptation of post-synaptic activity.
The model explains the behavior of cortical neurons by a combination of long-term potentiation and log-term depression given by a series of stimuli applied to pre-synaptic neurons [scholarpedia_].
Starting from the Hebbian learning rule, which established that repeated and persistent activities could determine a transmission of information between neurons, BCM model aims to overcome mathematical issues related to stability and applicability of perceptron models.

In this work we refer to the BCM implementation proposed by Law and Cooper in 1994 [pnas_], which is described by the set of equations:

.. math::

  \begin{align*}\label{eq:bcm}
    y               &= \sigma \left(\sum_i w_i x_i\right) \\
    \frac{dw_i}{dt} &= \frac{y (y - \theta) x_i}{\theta}  \\
    \theta          &= \mathop{\mathbb{E}}[y^2]           \\
  \end{align*}

where :math:`y_i` and :math:`\sigma` are the post-synaptic activity of the :math:`i`-th neuron and a non-linear activation function, respectively.

Shouval et al. [shouval_] proved the high selectivity of artificial neurons trained by BCM equations: synaptic connections tend to produce highly oriented receptive fields during the training, making neurons responsive to only a subset of provided patterns.
Several authors extended these results also to network architectures of BCM neurons [kirkwood_, blais_], highlighting the presence of receptive fields in neurons' synaptic.

Castellani et al. [castellani_] proposed to extend the classical BCM model including lateral connections between neurons.
Lateral connections between BCM neurons would allow to inhibit/increment the post-synaptic activities in relation to the state of neurons' neighborhood.
In other words, it involves the introduction of an extra matrix term (:math:`\mathcal{L}`), which influences the post-synaptic vector as

.. math::

  \begin{equation}
    \mathbf{y} = \sigma \left((1 - \mathcal{L})^{-1} W X \right)
  \end{equation}

where :math:`W` and :math:`X` are the synaptic weights matrix and the input matrix, respectively.
The introduction of lateral connections determines the selectivity level of the BCM neurons.
Inhibitory lateral connections would tend to discourage neurons from memorize the same patterns, while positive lateral connections increment the probability to reach the same stationary state (convergence) by several neurons.
Therefore, the strength of lateral interaction directly determines the learning capacity of the model.

A complete documentation about the mathematical background of the BCM model can be found here_.

.. _scholarpedia: http://www.scholarpedia.org/article/BCM_theory
.. _here : http://www.scholarpedia.org/article/BCM_theory
.. _pnas : https://www.pnas.org/content/91/16/7797
.. _shouval : https://pubmed.ncbi.nlm.nih.gov/8697227/
.. _kirkwood : https://www.nature.com/articles/381526a0
.. _blais : https://link.springer.com/chapter/10.1007/978-1-4757-9800-5_41
.. _castellani : https://iopscience.iop.org/article/10.1088/0954-898X/10/2/001