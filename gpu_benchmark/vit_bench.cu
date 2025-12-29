
/*
1) BENCHMARK: How much time I need to load the model? 
- Create the class + allocate + load the weights + create the descriptors

2) BENCHMARK: How much time for a single inference pass?

3) BENCHMARK: How much time takes for the model to process a dataset?

4) BENCHMARK: How much time for a multi stream vs a single stream inference pass?
*/

//Do WARM UP and then  N runs to take the times 
//1) Each run only create and loads the model (1 time maybe more)
//2) Each run takes the time to load the data and process it (2 times)
//3) Each run takes the total time from the first load of the batch to the final classification output (1 time)

/*
Parameters:
- number of streams
- batch
- minibatch 
- patch embedder:
    transpose stride
    position embeddings add stride
- block:
    residual stride
- layer norm:
    tokens per block
    channels stride value
- mlp bias stride
*/