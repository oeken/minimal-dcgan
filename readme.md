# Minimal DCGAN in TensorFlow
This is an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) in [TensorFlow](http://tensorflow.org). The example is based on the original work from [Radford et. al. ](https://arxiv.org/abs/1511.06434), where the generator learns to reconstruct samples from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

**Tested on:**  

- *macOS* - *CPU* - *Python 3.5* - *Tensorflow 1.1.0*
- *ubuntu xenial* - *GPU* - *Python 3.5* - *Tensorflow 1.1.0*

# Example Outputs
Digit generations are as following after *5000* and *20000* iterations:

<img src="models/dcgan_z50_2/res/0005000.png" width="480">
<img src="models/dcgan_z50_2/res/0020000.png" width="480">

# License

Copyright (c) 2018, M.O. Eken All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.