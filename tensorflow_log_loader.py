import numpy as np
from event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    # training_accuracies  =  event_acc.Scalars('training-accuracy')
    # validation_accuracies = event_acc.Scalars('validation_accuracy')

    accuracy = event_acc.Scalars('accuracy')

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = accuracy[i][2] # value
        # y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='val accuracy')
    # plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    # log_file = "./logs/events.out.tfevents.1456909092.DTA16004"
    log_file = "/home/kang/Documents/work_code_PC1/pt_deepglobe_challenge/logs/run_7/validation/events.out.tfevents.1523273177.TUBVLMF-fuerst"
    plot_tensorflow_log(log_file)







