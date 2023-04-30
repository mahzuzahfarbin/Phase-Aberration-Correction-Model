# AI-registration-pipeline
This repository will contain PyTorch + TensorFlow files using Python 3.6 to develop an AI registration pipeline for phase aberration correction in CT scans. Furthermore, an in-depth documentation for ML implementations are provided.

# Blood Spinal Cord Barrier
The blood spinal cord barrier (BSCB) is a physical barrier between the blood and spinal cord. The BSCB prevents toxins, blood cells, and pathogens from entering the spinal cord - all while maintaining a tightly controlled chemical balance in the spinal environment. However, through the application of focused ultrasound and its interaction with microbubbles can non-invasively open the BSCB for the therapeutic delivery of drugs.

![github-small](https://user-images.githubusercontent.com/117220797/235368837-2b0dad80-95e6-48ff-90ef-b55c57c5d08b.png)

*An image of the biological geometry of the Blood Spinal Cord Barrier (BSCB).*

# Phase Aberration Correction
**Aberation deals with the distortion or deviation of a wavefront from its ideal shape as it propagates through a medium.**

One of the main sources of image degradation in ultrasound imaging is the phase aberration effect, which imposes limitations to both data acquisition and reconstruction. 

*Background*: For treating spinal cord disorders, aberration refers to the changes that occur to the ultrasound beam as it passes through the vertebral bone. This is because vertebral bone has a different acoustic impedence compared to surrounding tissue, which causes the ultrasound beam to scatter as it passes through the bone. This scattering and reflection causes the ultrasound beam to distort, resulting in a non-unform distribution of energy within the focal region. 

In order to overcome aberration, correction algorithms are applied to the transducer array to modify the phase and amplitude of the ultrasound waves, so that they can converge at the desired location and intensity within the spinal cord. These algorithms use computational models that take into account the acoustic properties of the vertebral bone and geometry of the transducer array to calculate the necessary corrections.

![github-small](https://user-images.githubusercontent.com/117220797/235369264-52d2c0d0-a6f4-437c-b176-8ed7d9a7db51.png)

*An image of the phase aberration correction required in ultrasound images (numerical example).*

# Correction Algorithms
The most common computational model for phase correction is based on the K-Space method. This method uses the Fourier transform of the ultrasound pressure wavefront. It allows for the simulation of the propagation of ultrasound waves through complex media, such as the bertebral bone, and the calculation of the wavefront distortions caused by the bone. The K-Space method can be combined with optimization algorithms to determine the optimal corrections for aberration.

***Fourier Transform***

Fourier transform is a mathematical operation that decomposes a function or signal into its constituent frequencies. Any complex waveform can be expressed as a sum of simple sin(x) waveforms of different frequencies, amplitudes, and phases. We can then analyze its spectral content and identify important features such as peaks, harmonics, or noise. This is done via an algorithm called the Fast Fourier Transform (FFT). 

In focused ultrasound, FFT is used to analyze the waveform of the ultrasound beam and calculate the optimal phase and aplitude corrections needed to correct for aberration caused by the vertebral bone. FFT decomposes wavefront into its frequency components and then adjusts the phase and amplitude of each component for the desired focal point.

*Note: There are other computational models, but they tend to have a worst time complexity in comparison to K-Space.*

# K-Wave
K-Wave is a MATLAB toolbox that is designed for the simulation of acoustic wave properties (e.g., reflection, refraction, absorption, and effects of bone on ultrasound waves) and scattering in heterogeneous media, including soft tissue and bone. This toolbox is based on the K-Space pseudospectral method described in the previous section.

# Ray Acoustics
Ray acoustics is a modelling approach that is used to simulate the propagation of acoustic waves in complex media. In the context of focused ultrasound-mediated opening of the BSCB, ray acoustics is used to simulate the propagation of the ultrasound waves through the tissue, taking into account the effects of refraction, reflection, and absorptions, and the effects of the vertebral bone on the ultrasound wavefront. The simulations can be used to optimize the design of the transducer array and the treatment parameters.

# Deep Learning Approaches
Machine learning is a type of artificial intelligence that involves training an algorithm to make predictions or decisions based on input data. In machine learning, the implemented algorithm is dynamically programmed such that it is able to learn patterns on input data sets and subsequently apply such predictive and decision-making techniques on new, unseen data. 

Neural networks are a type of machine learning algorithm that are modeled after the structure and function of the human brain. They are composed of layers of interconnected neurons that process and transform input data to produce output. The neurons in each layer receive inputs from the previous layer, process them using a set of weights and biases, and then output their activation to the next layer. The output of the final layer is the network’s prediction or classification of the input data. One significant importance of neural networks is their effectiveness as image recognition; in particular, they are able to learn complex patterns and relationships in imaging data and make accurate classifications necessary from these datasets, even if these inputs contain noise or are incomplete.

Generative Adversarial Networks (GANs) are a type of neural network that are used for generating new, synthetic data samples that are similar to data from a given dataset. GANs are composed of two neural networks: a generator and a discriminator. The generator network takes in random noise as input and generates new data samples (e.g., images, sounds, or text). The discriminator network takes in both the generated data and real data from a dataset, and classifies which samples are real and which are not. As the two networks are trained together, the generator learns to produce more realistic data, while the discriminator learns to better distinguish between real and fake data.

U-Net GANs are a type of Generative Adversarial Network that uses a U-Net architecture as the generator network. U-Net is a fairly popular neural network architecture for image segmentation tasks, and has been adapted for use in GANs to generate synthetic images. 

In a U-Net GAN, the generator network takes in a random noise vector as input and generates a synthetic image as output. This architecture consists of an encoder and decoder, which are connected by a bottleneck layer. The encoder down-samples the input image, while the decoder up-samples the output image, with the bottleneck layer in between to capture the most important features of the image. 

Furthermore, in a U-Net GAN, the discriminator network is a convolutional neural network (CNN) that is trained to distinguish between real and synthetic images. The discriminator is designed to be more complex than the generator, and is trained to identify the subtle differences between real and synthetic images. 

# Project Outline
Treatment planning using computational models and patient-derived geometry and bone density can calculate the necessary corrections to be applied to the transducer array. Unfortunately the registration and simulation pipeline can take several minutes, or longer for more accurate models, to perform these calculations, which is cumbersome in the treatment setting. In the present project we will investigate the use of machine learning methods to accelerate both the registration and aberration calculation steps, leveraging a large spine CT dataset containing images from over 100 patients. The ultimate impact of this work will be the development of practical tools that will streamline the workflow of large animal preclinical safety studies and ultimately of the clinical studies.
