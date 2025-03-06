# NeuralVelocimetry
Implementation of the paper **Image Velocimetry using Direct Displacement Field estimation with Neural Networks for Fluids** [[preprint]](https://arxiv.org/abs/2501.18641).

Where a neural network is used to aproximate the displacement field between a pair of consecutive images.


# How to install
First clone the repository.
Then you need to install PyTorch, following the instructions that they give on their page: [PyTorch Main Page](https://pytorch.org/).

Then you can install the requirements:
```shell
 pip install -r requirements.txt
```

# Usage
See [examples](examples/).

# Using the Gradio App
To use the Gradio app, additionaly from the previuos installed libraries, you have to install the [latest version of gradio](https://www.gradio.app/guides/quickstart#installation).

Then execute the **app.py** file on the DDFNN directory. This will give you a url where you can acces the app.
