
# LSTM language modeling on Penn Treebank dataset

This example is mainly to demonstrate:

1. How to train an RNN with persistent state between iterations.
	 Here it simply manages the state inside the graph. `state_saving_rnn` can be used for more complicated use case.
2. How to use a TF reader pipeline instead of a DataFlow, for both training & inference.

It trains an language model on PTB dataset, basically an equivalent of the PTB example
in [tensorflow/models](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)
with its "medium" config.
It has the same performance & speed as the original example as well.
Note that the data pipeline is completely copied from the tensorflow example.

To Train:
```
./PTB-LSTM.py
```


