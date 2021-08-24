import matplotlib.pyplot as plt
import numpy as np

def plot_dev_loss(title, epochs, losses, fname):
    # T5 fine-tuning
    plt.plot(epochs, losses)
    plt.ylabel('Cross entropy loss')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.savefig(fname)
    plt.close()
    #plt.show()

def t5_fine_tune():
    y = val_losses = [0.275, 0.199, 0.156, 0.1442, 0.1395, 0.1325, 0.1276, 0.1283, 0.1314, 0.1308]
    x = np.arange(len(val_losses))
    title = 'T5 fine-tuning dev loss (average F1)'
    plot_dev_loss(title, epochs=x, losses=y, fname='t5_fine_tune_dev_epochs.png')

def bart_fine_tune():
    y = val_losses = [0.4017, 0.2714, 0.2216, 0.1986, 0.2031, 0.2055, 0.2015, 0.2088, 0.2212, 0.2316]
    x = np.arange(len(val_losses))
    title = 'BART fine-tuning dev loss (average F1)'
    plot_dev_loss(title, epochs=x, losses=y, fname='bart_fine_tune_dev_epochs.png')

def bart_without_pretraining():
    y = val_losses = [0.5309, 0.5235, 0.5201, 0.5198, 0.5129, 0.5303, 0.5165, 4.9649, 2.5038, 5.0153]
    x = np.arange(len(val_losses))
    title = 'BART without pre-training dev loss (average F1)'
    plot_dev_loss(title, epochs=x, losses=y, fname='bart_without_pre_training_dev_epochs.png')

if __name__ == '__main__':
    t5_fine_tune()
    bart_fine_tune()
    bart_without_pretraining()
