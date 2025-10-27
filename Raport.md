Homework nr. 2


First experiment

For the first experiment I did the following things:

1.Used seed for reproducibility
2.Using early stopping during training
3.Added a learning rate scheduler: ReduceLROnPlateau
4.Added checkpoint saving utility
5.USED ADAM OPTIMIZER, to see how predictions improve
6.Did minimal data augmentation: Used mirroring

Best score for the first experiment: 0.64750
Running time for the training loop: 

Second experiment

For the second experiment I did the following things:

1.Added these data augmentation techniques: RandomCrop,RandomHorizontalFlip,RandomAffine,ColorJitter
2. Because the Adam optimizer improved my score in the first experiment, I decided to keep it for the second experiment to see how data augmentation changes my results.
3.Changed LRScheduler to CosineAnnealing
4.Smoothed labels
5.Added epochs

Best score for the second experiment: 
Running time for the training loop: 


Third experiment

For the third experiment I did the following things:


1. I used CutMix and Mixup 
2. I kept the rest of the configuration the same

Best score for the second experiment: 
Running time for the training loop: 


Fourth experiment

For the fourth experiment I did the following things:

1. Configuration is the same as first experiment, but i used this optimizer: SGD but with momentum.

Best score for the second experiment: 
Running time for the training loop: 


