# Cellular Automata

We use cellular automata to construct a simple algorithmic learning task dependent on data ordering.
We show that by manipulating the ordering of data in these tasks based on correlations between the underlying data points, we can speed up model convergence.
Crucially, measures which improve long term model performance (measured by decreasing loss) result in worse short term performance.

## CellPyLib

We use the CellPyLib python library for our simulations. However, we use a small number of functions and these could likely be written with little effort if this library presents difficulties.
The primary advantage is easy access to plotting software for the automata. Good summaries of these elementary cellular automata can be found in Stephen Wolfram's 'A New Kind of Science'.

## Task Breakdown

An elementary cellular automata is randomly initialised and then evolved for 100 time steps (chosen as an arbitrary but sufficiently high number). 
The automata is taken to be 100 cells long and acting under periodic boundary conditions.
The final state of the automata is used as the network input, and the classification variable is the value from 0 to 255 giving the rule of the automata. It is a multiclassification task.

This structure clearly lends itself well to a task breakdown, as each individual automata is it's own task. 
The difference between this and the sparse parity work is that here sparsity is entirely irrelevant due to the translational symmetries in the problem.
Also, correlations can be calculated between tasks and used to inform data ordering, rather than arbitrary probabiility distributions.

The image below gives an example of what the accuracy curves often looked like. 
They are quite unclear, and for this reason often only a few tasks are viewed or only certain runs are focused on in order to find interpretable signals.
Regardless, the findings turn out to be quite reproducible.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/648b9157-5931-48d4-801e-fcdf7d3e7d82)

## Data Ordering

We impose a probability distribution over the tasks representative of some underlying feature we wish to study.
In this instance, all data is algorithmic and so can be generated at will. The intention is to use this artificial space to study the impact of ordering real work data on network performance.
This work could be considered an aspect of curriculum learning.

The most obvious way to order data, and the method used in much of the literature, is to start with 'easy' tasks and move on to 'hard' tasks.
We first test this in the most obvious way, by training a large model over all tasks and taking the tasks on which it has the highest accuracy to be the easiest.

The below image shows the result of this test, making 'easier' tasks more frequent.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/854b34e2-8c91-4f94-a503-134f54dfb2e2)

And below you can see the average accuracy.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/907a36eb-b809-4f2e-8551-294d644e63db)

We can see that by manipulating the ordering of data some strange things can be induced. 
Notably, even though loss continually declines throughout this training run, accuracy decreases after a certain point when the easy tasks have been forgotten to accomodate harder tasks.

## Correlation

Correlation between tasks can be calculated, and then tasks ordered by this correlation.

When tasks which are more correlated with themselves (higher autocorrelation) are used, learning is faster initially but saturates more quickly.

The image below shows the case when tasks with a higher autocorrelation are made less frequent in the training data set.
Such that the least autocorrelated tasks come first.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/38b7d38e-39e2-4ca1-8860-099c2d97b964)

It can be seen that in this instance, learning is slower initially but the maximum accuracy for a given model size is higher. 

The corelation matrix between tasks lends itself to a hierarchical clustering, which has more clear results.
When tasks higher in the hierarchical clustering (which are colloquially 'more correlated' with other tasks) are made more frequent, learning occurs quickly but only for those tasks.
This can be seen in the image below.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/97cbc423-67e4-4327-a118-e352d0f4de3c)


When the opposite is done, learning is much smooth, more evenly distributed across the tasks, and generally more effective.
Tasks no longer see wild variation in their accuracy, seeing more gradual learning implying some numerical instabilities may have been responsible for earlier failures.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/1aee0a28-aaae-4413-80b3-e92d5fed09c7)


These findings reproduce well across different runs. However, there can be substantial variation in the order in which different tasks are learned.

## Applications

We believe that these ideas will be useful in deep learning applications using non-algorithmic data. 
It shows how when the correlations between tasks are accounted for, learning can be made far quicker and more stable by carefully setting the order in which data is shown.
