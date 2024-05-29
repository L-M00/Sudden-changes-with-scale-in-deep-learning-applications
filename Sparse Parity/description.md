# Sparse Parity

The sparse parity experiments are the most clear findings of the work, and contain many of the clear effects searched for.
By constructing a machine learning task such that one part of the input is much more 'relevant' to the learning than the rest, very sharp learning behaviour can be seen epochwise.
The first few bits (between 3-8 in the experiments shown in the dissertation) contain information about which task should be performed. The rest of the input, termed the 'message', has that task performed on it.
Therefore, for minimising loss the first few bits are by far the most important.

## Under- and Over-Parameterisation

Below there is an example of learning on a sparse parity task with 10 different tasks and a comparably small network. 
It can be seen that, unlike the case when the network is clearly overparameterised, the learning happens in stages. 
Each task is still learned suddenly (in a manner termed a 'sigmoidal emergence'), but overall the tasks are not all learned at once. Nor is there a clearly transition in the network which makes sparse learning possible.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/27c3dc40-4682-47d5-abd6-fa7b5c02c1c4)

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/72af76cb-9c9a-4040-9bca-9d6855b5c3b6)

When we transition to a more overparameterised network (notably, increasing the sparsity helps this also), we get much more sharp learning. 
In the image below, we have only used a single task to emphasise this effect. However, it is a general property of these systems that the more overparameterised they become the sharping this learning becomes.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/8148e37f-4972-4d01-a6b6-bef6def0ad99)

Although, addition far more tasks and just barely overparameterising the network removes some of this, as can be seen in the figure below. 
It seems that the overparameterisation is not just a function of the number of data points, but also the complexity of the underlying tasks.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/7cdad084-c4c8-4621-8ec4-e074cb839411)

## Probability distributions

The next step is to introduce a probability distribution over tasks, to test how this impacts long run learning.
The image below shows the impact of weighting some tasks just slightly below others. It can be seen that even for a large (but not overwhealmingly overparameterised) network, relativity frequency still really matters.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/1ab777be-1936-429a-979f-34ebaeffc8bf)

Even though very rare tasks are still learned well, they still consistently lag others.
This is to be expected for a network with limited capacity and data, to not focus overtly on rare data. It is possible that a technique similar to class weighting would aid this.

## Train-Test Gap

As can be seen in the above images, there is a clear gap between the sigmoidal emergence in train and in test accuracy and loss.
While this gap is rarely very large, the fact that train accuracy can sometimes reach as high as 80% before test accuracy has gone above 50% implies an interesting effect.
There is a clear connection to grokking, the case where this gap can be systematically increased greatly.

Most relevant figures for this can be found in the dissertation text. 
It is relevant that the one alteration that consistently increased this train-test gap was the message length, indicating that a sufficiently sparse task will experience full grokking.

## Infinite data

Data is generated at each epoch for training. While this is not useful for studying properties such as the train-test gap, it is useful for bypassing computational memory constraints.
Models trained on millions of data points can be constructed, allowing more in depth understanding of these systems.

Consider the difference between the image below, created in a finite data regime with a more traditional train and test accuracy and loss distinction.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/8be5e2a2-3000-4002-9114-0b71580b0837)

And the following image, created under the infinite data regime.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/8b902537-e646-4f0b-af76-c4f260767d7f)

It can be seen that the plateau, indicative normally of model saturation in the first image, was in fact only due to data constraints.
This will be very relevant when considering transfer learning, for which we will need extremely large amounts of diverse data.

## Transfer Learning

We consider the connection between transfer learning and sigmoidal emergences to be the most important finding in the dissertation.
Transfer learning refers to the ability of a model to generalise to previously unseen tasks due to related training.
By maintaining the same sparse structure, which I am defining as the position of the task code and the message, but randomising the tasks, model transfer is highly successful.

The image below shows the task accuracy when training a model on a set of tasks, such that each task appears with proability $\frac{1}{2}$ compared to the previous one.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/567b687c-2d35-415d-971b-193514ed0de9)

It takes around 5000 epochs to learn anything, and then achieves around 99% accuracy in just 20 epochs.
We then use the same model and sparse structure, but completely randomise the tasks being learned.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/a7cb79bb-d604-476f-afdc-19c2ac7a17cb)

This time, the learning only took around 20 epochs. Again, the accuracy reached around 99% in that time. 
The implies that during the preceeding 5000 epochs all that was being learned was the sparse structure. 
Notably, the loss does consistently decline in this slow flat period, but does so extremely slowly. 

The number of epochs required for this sigmoidal emergence (or sudden increase in accuracy) is somewhat predictable.
It has a random component due to initialisation, but consistently lies within a broad band of number of epochs. 
It can be manipulated by altering certain features of the data and the network itself. 
The findings from the train-test gap section regarding the positions and nature of sigmoidal emergences apply well to the infinite data regime, even though there is no longer a distinction between train and test.
