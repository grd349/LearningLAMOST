import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_results(folder):
    conf = np.loadtxt('Files/' + folder + '/confusion.csv', delimiter = ',')
    accuracies = np.loadtxt('Files/' + folder + '/accuracies.csv', delimiter = ',')
    
    classes = np.loadtxt('Files/' + folder + '/classes.csv', dtype = 'str', delimiter = ',')
    
    cls = len(classes)
    
    fig, ax = plt.subplots(2,2, figsize=(18,12))
    
    fig.suptitle('Neural Network Classification of Black Body Spectrum')
    
    #colours = ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'orange', 'yellow', 'teal', 'light green']
    
    colours = ['navy', 'royal blue', 'blue', 'cerulean', 'sea blue', 'turquoise', 'aqua', 'cyan', "robin's egg blue", 'pale blue', 'green', 'orange', 'red', 'yellow', 'blue']
    
    actual = conf.copy()
    
    for i in range(cls - 1):
        actual[i+1] = actual[i+1] + actual[i]
    
    for i in range(cls)[::-1]:
        print(i)
        bar = sns.barplot(classes, actual[i], ax = ax[0][0], color = sns.xkcd_rgb[colours[i]], label = classes[i])
    
    ax[0][0].set_title('Correct Classification of Spectra')
    ax[0][0].set_xlabel('Actual Classification')
    ax[0][0].set_ylabel('Amount')
    ax[0][0].legend(title = 'Model\nClassification', framealpha = 0, bbox_to_anchor=(0.99, 0.9))

    for b in bar.patches[cls-1:-1:(cls-1)]:
        b.set_hatch('//')
    
    ax[0][1].plot([a[0] for a in accuracies], [a[1] for a in accuracies])
    ax[0][1].set_title('Accuracy of Model with number of Training Cycles')
    ax[0][1].set_xlabel('Training Cycles')
    ax[0][1].set_ylabel('Model Accuracy')

    model = conf.copy().transpose()
    
    for i in range(cls - 1):
        model[i+1] = model[i+1] + model[i]
    
    for i in range(cls)[::-1]:
        bar = sns.barplot(model[i], classes, ax = ax[1][0], color = sns.xkcd_rgb[colours[i]], label = classes[i])
    
    ax[1][0].set_title('Model Classification of Spectra')
    ax[1][0].set_xlabel('Amount')
    ax[1][0].set_ylabel('Model Classification')
    ax[1][0].legend(title = 'Actual\nClassification', framealpha = 0, bbox_to_anchor=(0.99, 0.9))
    
    for b in bar.patches[cls-1:-1:(cls-1)]:
        b.set_hatch('//')

    correct = 0
    total = 0
    
    for i in range(cls):
        correct += conf[i][i]
        total += np.sum(conf[i])
        
    wrong = total - correct
    
    ax[1][1].pie([correct, wrong], labels = ['Correct', 'Incorrect'], autopct='%1.2f%%')
    ax[1][1].set_title('Overall Accuracy')
    ax[1][1].axis('equal')
    
    plt.savefig('Files/' + folder + '/results_' + folder + '.pdf')
    plt.show()

#possible additional/alternative plots
#plot of classification against some feature (i.e. temp for black bodies)
#pie charts for each classification

if __name__ == "__main__":
    plot_results(sys.argv[1])
