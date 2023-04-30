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

# K-Wave - MATLAB Toolbox
