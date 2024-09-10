# RNN Project Overview

This project was primarily developed to deepen my understanding of Recurrent Neural Networks. In the `model.py` file, there are two types of models implemented:

1. **PyTorch-based RNN models**: Utilizing PyTorch's built-in RNN modules.
2. **RNN from scratch**: Using pytorch module nn.Linear and nn.Embedding.

The objective was to compare the training efficiency between these two approaches. Unsurprisingly, the PyTorch-based models are significantly faster, and I don't understand exactly why.

## Sampling from the Shakespeare Dataset

Here are some samples generated by the models:

*ERBY:*
*A dod of all good damned safely,*  
*Which have I live, prithts are unto his princely hence,*  
*We will dry 'like something but a dead death?'*

*DUKE VINCENTIO:*  
*I would thou shalt be you, the footman-blood upon*  
*Is law a sea till do we are with the*

## Usage Instructions

### Training the models
To train the models, use the following bash command:

```bash
python train.py --config $configPath
```
