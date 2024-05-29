The Mandelbrot set is a classic example of a fractal, showing high levels of complexity regardless of the length scale. A low resolution plot can be seen below.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/b2b8dbc1-40ee-4ce2-a880-cdcb99eb0083)

For our purposes, we will be considering only the boundary of the Mandelbrot set up to some given zoom. 
By using a greedy algorithm, checking every point in a given lattice domain, we find the number of steps required for an orbit to reach $|z|=2$, at which point it is no longer in the set.
If a given lattice point very nearly escapes before some threshold number of iterations, it is designated to be on the boundary.
Points on the boundary are classified as 1, and points not on the boundary are classified as 0.
Below shows the loss of a model trained on such data. Note that new data was generated for each training iteration, and data ordering was used to speed up convergence.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/24bf6b9e-44f9-47c0-b218-7af1754dd300)

As can be seen, there is a sudden drop during training for a suitably overparameterised model with certain kinds of data ordering.
Below, we show the decision boundary of this network, showing a clear outline of the Mandelbrot set as found in the discretised training data.

![image](https://github.com/L-M00/Sudden-changes-with-scale-in-deep-learning-applications/assets/125475518/7378e5f6-1b00-42fc-934f-02da52d37392)

Note that the code here as well as associated descriptions did not enter the dissertation. 
However, the ideas relating to data generation and ordering to test curriculum learning and its impact on sigmoidal emergences inspired the more clear cellular automata work.
