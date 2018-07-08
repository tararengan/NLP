# NLP
Standalone repo for text generation, language processing etc.
The file textgenerator.py is a slight variant of Andrej Karpathy's implementation at https://gist.github.com/d4dee566867f8291f086.git. It repurposes forward pass to calculate derivates.
It also includes derviates w.r.t. the last hidden layer of the previous batch (except when the current batch is a reset on the input text.
