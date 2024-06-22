# TnC-reader
## Inspiration
I have always been interested in AI and machine learning, and I realized that many people don't read the Terms and Conditions on websites because they are so long and convoluted. I wanted to fix that be summarizing it into a few paragraphs, so people could figure out what they were agreeing to.
## What it does
The code launches a desktop application that contains two text boxes, input and output, and two buttons to summarize and clear the text.
## How we built it
The code is made in Python, and uses the PyQt6 library to create the desktop application. It uses Facebook's Bart Large CNN model from HuggingFace to do the tokenization and summarization. It runs in a separate thread to avoid freezing the main thread.
## Challenges we ran into
In the beginning, the AI processing occurred extremely slowly (over 5 minutes for one sentence). However, I realized that it was running on the CPU instead of using CUDA and running on the GPU. Another challenge that still hasn't been fixed is that I had make a tradeoff between the maximum length of the input text and the speed of summarization. I considered using Google's Pegasus X, but I tried it and summarization was also extremely slow, even when using CUDA. I ultimately decided to stick with Bart, but I would like to find a faster model or make some optimizations in the future.
## Accomplishments that we're proud of
I am extremely proud of how much I learned throughout the project. I learned PyQt6 from scratch as I had never made desktop applications in Python. In addition, I had never used HuggingFace or used Machine Learning in this scenario.
## What we learned
I learned a lot about PyQt and Machine Learning throughout this project. In particular, I started off knowing nothing about LLMs, but now I have a basic understanding of how they work, with tokenization and summarization.
## What's next for T&C Reader
I plan to work on it for a little while in the future. I hope use a different model that can handle a larger input and optimize it to work in a reasonable amount of time. I would also like to improve the UI and dynamically retrieve the Terms and Conditions from a website given the URL.
## How to Run
If you want to use CUDA, you can install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), then install [PyTorch](https://pytorch.org/get-started/locally/) with the correct CUDA version.
```bash
$ python main.py
```
