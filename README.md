# ray
Building a Neural Network with "Inventing on Principle" in mind

2020-05-17

In the last years I spent some time working with Neural Networks and I never really got over the point of trying stuff out in the hope it works and often it did not. After a while it became frustrating and progress seemed like winning the Lottery. How can I improve the network when I barely understand it? Sure I learned the math behind it, but after a week I forgot half of it and the matrices make the whole matter rather abstract. I am a Software developer and I want to see whats happening in my software, not let it run for 10 hours in a quasi black box to get some result I can just hope to have a good result. Is it possible to really debug deep learning? I dont know, but I want to try.

The long term plan is do visualize a lot and try different things. Maybe changing the size of layers dynamically? That would be something hard to realize with matrix calculations. In this program every cell and every connection has complete responsibility for itself and all it knows are the connections from coming from the cells before and the connections going out to the next cells. The hope is that an approach with extreme multi threading is possible, so in the extreme case every cell would have its own thread.

For now its all just trying for fun.
