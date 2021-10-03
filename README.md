# Mini-inception


---

This is an implementation of a simple version of the famous `Inception` neural network for object recognition.

The implementation is done with the model subclassing API. Tensorboard is used also to allow a nice monitoring of the results


To run the code, just write on a bash shell the following command : 

```bash
python3 main.py --batch-size=<your_value> --epochs=<your_value> --buffer-size=<optional_argument> --freq-display-train=<your_value> --freq-display-val=<your_value>

```

The arguments we have : 

- batch-size : batch of data to use to update weights. Note that the optimizer used is Adam.
- epochs : number of epochs
- buffer-size : Affects the randomness (order) in which the elements are ordered in each batch. You don't have to touch it it is set by default to 1024.
- freq-display-train : frequency of printing training loss values, default to 20
- freq-display-val : frequency of printing validation loss values, default to 40


Once the training is done, you can run the following command to lauch the web application and visulaize the traininig elements.

```bash

tensorboard --logdir tmp/tb-logs
```

