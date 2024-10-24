import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

mpl.use('Agg')

class Visualization():
    def __init__(self, time, model_type, step):
        """Initializes visualization.
        """
        # set attributes
        self.time = time
        self.model_type = model_type
        if 'ElemPred' in model_type:
            self.model = 'element prediction'
        elif 'EmpPred' in model_type:
            self.model = 'empowerment prediction'
        else:
            self.model = 'link prediction'
        self.step = step

        # set general settings for plotting
        # TODO: change font to Open Sans
        sns.set_theme(context='paper', style='ticks', font='Arial', font_scale=2.5, rc={'lines.linewidth': 2,
                                                                                        'axes.linewidth':0.6, 'axes.edgecolor': '#9d9d9d'})
        self.colors = ['#0c2e8a', '#ffc640']
        sns.set_palette(sns.color_palette(self.colors))

    def plot_progress(self, history, metrics):
        """Plots training progress and saves file as PNG.
        """
        # print info
        print('\nPlot training progress.')

        # set figure size
        plt.figure(figsize=(12,10))

        # plot data
        for n, metric in enumerate(metrics):
            name = metric.replace('_',' ')
            if name == 'auc':
                name = 'AUC'
            elif name == 'mse':
                name = 'MSE'
            elif name == 'mae':
                name = 'MAE'
            elif name == 'cosine similarity':
                name = 'Cosine similarity'
            else:
                name = name.capitalize()

            # plot data
            plt.subplot(2,2,n+1)
            plt.tight_layout()
            plt.plot(history.epoch, history.history[metric], label='train')
            plt.plot(history.epoch, history.history['val_'+metric], linestyle='--', label='val')

            # set title, labels, legend
            # plt.title('Training progress for {} model, metric={}'.format(self.model, metric), loc='center', wrap=True)
            plt.xlabel('Epoch', fontsize=24)
            plt.ylabel(name, fontsize=24)
            plt.legend()

        # save plot
        plt.savefig('data/gametree/{}/{}-round{}-trainProgress.png'.format(self.time, self.model_type, self.step))
        plt.savefig('data/gametree/{}/{}-round{}-trainProgress.svg'.format(self.time, self.model_type, self.step))
        plt.close()

    def plot_confusion_matrix(self, labels, predictions, p=0.5 ):
        """Plots confusion matrix for train and test set.
        """
        # print info
        print('\nPlot confusion matrices.')

        # set figure size
        plt.figure(figsize=(12,6))

        # plot confusion matrices
        for i, set_type in enumerate(['train', 'test']):
            # plot data
            plt.subplot(1,2,i+1)
            plt.tight_layout()
            c_m = sns.light_palette('#0c2e8a', as_cmap=True)
            cm = confusion_matrix(labels[i], predictions[i] > p)
            sns.heatmap(cm, annot=True, fmt='d', cmap=c_m)

            # set title, labels, legend
            plt.title('{} set '.format(set_type.capitalize()), loc='center', wrap=True)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')

            # print info for user
            print('\t{} set:'.format(set_type))
            print('\t\ttrue negatives: ', cm[0][0])
            print('\t\tfalse positives: ', cm[0][1])
            print('\t\tfalse negatives: ', cm[1][0])
            print('\t\ttrue positives: ', cm[1][1])

        # save plot
        plt.tight_layout()
        plt.savefig('data/gametree/{}/{}-round{}-confusionMatrix.png'.format(self.time, self.model_type, self.step))
        plt.savefig('data/gametree/{}/{}-round{}-confusionMatrix.svg'.format(self.time, self.model_type, self.step))
        plt.close()

    def plot_roc(self, labels, predictions):
        """Plots ROC curve for train and test set.
        """
        # print info
        print('\nPlot ROC curves.')

        # set figure size
        plt.figure(figsize=(12,10))

        # plot roc curves
        for i, set_type in enumerate(['train', 'test']):
            fp, tp, _ = roc_curve(labels[i], predictions[i])
            plt.plot(100*fp, 100*tp, label=set_type, linewidth=2)

        # set title, labels, legend
        plt.xlabel('False positives [%]', fontsize=32)
        plt.ylabel('True positives [%]', fontsize=32)
        plt.xlim([0,100])
        plt.ylim([0,100])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.title('ROC curves', loc='center', wrap=True, fontsize=24)
        plt.legend(fontsize=32)

        # save plot
        plt.tight_layout()
        plt.savefig('data/gametree/{}/{}-round{}-rocCurve.png'.format(self.time, self.model_type, self.step))
        plt.savefig('data/gametree/{}/{}-round{}-rocCurve.svg'.format(self.time, self.model_type, self.step))
        plt.close()

    def plot_ranks(self, *ranks):
        """Plots density of predicted ranks for true results and returns info on ranks.
        """
        # print info
        print('\nPlot rank results.')

        # set figure size
        plt.figure(figsize=(12,6))

        # plot data as histogram (density)
        for i, set_type in enumerate(['train', 'test']):
            # plot data
            plt.subplot(1,2,i+1)
            ax = sns.histplot(data=ranks[i], kde=True, log_scale=True)
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            plt.minorticks_off()

            # plot mean
            try:
                kdeline = ax.lines[0]
                mean = np.array(ranks[i]).mean()
                height = np.interp(mean, kdeline.get_xdata(), kdeline.get_ydata())
                ax.vlines(mean, 0, height, ls='dashed', color='#444444', linewidth=1)
            except:
                pass

            # set title, labels, legend
            plt.title('{} set'.format(set_type.capitalize()), loc='center', wrap=True)
            plt.xlabel('Predicted ranks for true results')
            plt.ylabel('Count')

        # save plot
        plt.tight_layout()
        plt.savefig('data/gametree/{}/{}-round{}-rankDensity.png'.format(self.time, self.model_type, self.step))
        plt.savefig('data/gametree/{}/{}-round{}-rankDensity.svg'.format(self.time, self.model_type, self.step))
        plt.close()

        # return info on test ranks
        return {'Mean': np.mean(ranks[1]), 'Std': np.std(ranks[1]), 'ProbTrueResult': ranks[1].count(1)/len(ranks[1])}
