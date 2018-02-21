import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class NN_Results():
    def __init__(self,folder):
        self.accuracy = np.genfromtxt('Files/' + folder + '/Accuracy.csv', delimiter=',')
        self.conf = np.genfromtxt('Files/' + folder + '/Confusion.csv', delimiter=',')
        self.labels = np.genfromtxt('Files/' + folder + '/Labels.csv', dtype=str, delimiter=',')
        self.create_dataframes()
        
    def create_dataframes(self):
        heat_dict = {'Prediction':[], 'Classification':[], 'Value':[]}
        bar_dict = {'Classification':[], 'Total':[], 'Incorrect':[], 'Percent_Correct':[]}
        for idx in range(len(self.conf[0])):
            bar_dict['Classification'].append(self.labels[idx])
            bar_dict['Total'].append(np.sum(self.conf[idx]))
            bar_dict['Incorrect'].append(np.sum(self.conf[idx])-self.conf[idx][idx])
            bar_dict['Percent_Correct'].append((self.conf[idx][idx]/np.sum(self.conf[idx]))*100)
            for idx2 in range(len(self.conf[0])):
                heat_dict['Prediction'].append(self.labels[idx])
                heat_dict['Classification'].append(self.labels[idx2])
                heat_dict['Value'].append(self.conf[idx][idx2])
        heat_df = pd.DataFrame(data=heat_dict)
        self.heat_df = heat_df.pivot('Prediction', 'Classification', 'Value')
        self.bar_df = pd.DataFrame(data=bar_dict)
        
    def plot_results(self):
        fig, ax = plt.subplots(2,2,figsize=(13,10))
        fig.suptitle("Convolutional Neural Net",y=1.02,fontsize=24)
        
        sns.set_color_codes("pastel")
        sns.barplot(x='Total', y='Classification', data=self.bar_df, label='Correct', ax=ax[0][0], color='g')
        sns.set_color_codes("muted")
        sns.barplot(x='Incorrect', y='Classification', data=self.bar_df, label='Incorrect', ax=ax[0][0], color='g')
        ax[0][0].set_xlabel("Number of Predictions")
        ax[0][0].set_ylabel("Classification")
        ax[0][0].set_title("Correctness of Predictions")
        ax[0][0].legend(ncol=2, loc="lower right", frameon=True)
        
        ax[0][1].plot(np.linspace(0,(len(self.accuracy)-1)*10,len(self.accuracy)), self.accuracy, c='g')
        ax[0][1].set_xlabel("Batch number")
        ax[0][1].set_ylabel("Accuracy")
        ax[0][1].set_title("Accuracy with Training Batch")
        
        sns.barplot(x='Classification', y='Percent_Correct', data=self.bar_df, ax=ax[1][0], color='g')
        ax[1][0].set_ylabel('Percentage / %')
        ax[1][0].set_title('Percentage of Correct Predictions')        
        
        sns.heatmap(self.heat_df, cmap="Greens", ax=ax[1][1])
        ax[1][1].set_title("Heat Map of Confusion Matrix")
        for tick in ax[1][1].get_xticklabels():
            tick.set_rotation(90)
        for tick in ax[1][1].get_yticklabels():
            tick.set_rotation(0)
        
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    results = NN_Results('DR3_2')
    results.plot_results()