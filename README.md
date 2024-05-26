# Sudden Changes with Scale in Deep Learning Applications

This repository contains code used in the dissertation 'Sudden Changes with Scale in Deep Learning Applications'. 
As entirely algorithmic datasets were used, this also includes the data generation.
All code used to generate figures is available, although in certain cases minor modification has been made to improve readability.
We have also included additional code used to inform our discussion where we did not include our full results. This notably includes the case where an autocorrelation ordering was used for cellular automata classification, and certain simple cases of overparameterised neural networks.

## Double Descent

Firstly, you can find code relating to double descent in linear regression, and how the effect can be better understood by considering particular quickly varying function classes. Some of this code is adapted from the code used in the paper https://arxiv.org/abs/2303.14151. 

We show that double descent is not purely an artifact of the regularisation at play, nor is it the case that the maximum of generalisation error always occurs at the interpolation threshold $N=P$. In the case where a function being fitted is well approximated by a single feature (as is the case in the left hand plot below, and not the case in the right hand plot) more complex behaviour can be displayed.

![Figure - Double descent pair linear scale 10 more repeats](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/eb2b5aa7-bc31-4008-a55e-7731d46bb71c)

## Sparse Parity

Using a sparse parity task, we construct a specific algorithmic dataset where learning takes the form of clear and well-defined sigmoidal emergences. That is, there is no learning for a lengthy period followed by rapid learning that seems to act as a linear function of the number of datapoints seen after some particular number of epochs.

By retaining the same sparse structure of the data, but randomising the specific tasks being used and retaining the same model, it can be seen that learning after that particular number of epochs is a function only of the number of epochs that have elapsed since the sigmoidal emergene. In the figure below, it can be seen that learning occurs immediately after the task reset. The model used was trained for approximately 7000 epochs with entirely different tasks (but retaining the same sparse structure), and for around 5200 epochs experienced almost no learning.

![Figure - Parity after transfer learning breakdown by task](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/0269ee95-f4ee-4bc3-bbd6-c1908b63e461)

## Cellular Automata

Using cellular automata as a toy machine learning problem, we show that by using correlation information between different tasks to order data, learning can be made more effective in the long run. By sacrificing short term decreases in loss, longer term learning can be improved by showing the network harder tasks first, avoiding local maxima. 

The below figure shows the effect of showing colloquially 'less correlated' tasks to the network more often than more correlated tasks. While it takes longer for loss to decrease initially, the loss can reach lower levels overall.

![Figure - 2x2 hierarhical, negative, moving_avg](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/420cf4ba-1172-4e94-a381-479af69ac6fe)

Note that the code used to perform these experiments and generate the relevant figures requires the cellpylib python library. Only a small number of such functions are used, and so with minor alterations to the data generation code this library requirement could be removed.
