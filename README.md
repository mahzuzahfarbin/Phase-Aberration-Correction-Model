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
