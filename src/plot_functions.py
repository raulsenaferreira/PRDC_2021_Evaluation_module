import os
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as ani
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from src import util
from sklearn.manifold import TSNE, Isomap
from keras.models import load_model
import seaborn as sns
#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon

sns.set()

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


set_tf_loglevel(logging.FATAL)


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    #usage
    #multipage('multipage_w_raster.pdf', [fig2, fig3], dpi=250)


def visualize_distributions(X, y, dataset_name, model_file):
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    X, y = unison_shuffled_copies(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    df_subset = {}
    number_of_classes = 10 # for visualization readability purposes it is limited to 10
    number_of_instances = 2000 # number of points to be ploted 

    #indices = np.unique(y, return_index=True)[1]
    indices = np.where(y < number_of_classes)[0]
    y = y[indices]
    y = y.flatten() # correcting bug
    y = y[:number_of_instances]

    # loading model and extracting the image features
    model = load_model(model_file)
    data = X[indices]
    data = data[:number_of_instances]
    
    #print(np.shape(data[:number_of_instances]))
    weights = util.act_func(model, data)

    #between 32 and 64 dim is enough for keep more than 90% of explained variance of the images:
    #https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_On_the_Intrinsic_Dimensionality_of_Image_Representations_CVPR_2019_paper.pdf
    isomap = Isomap(n_components = 32) 
    compressed_data = isomap.fit_transform(weights)

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(compressed_data)
    #tsne = TSNE(n_components=2).fit_transform(compressed_data)
    #print(tsne[:,0])
    #print(tsne[:,1])
    df_subset['one'] = tsne[:,0]
    df_subset['two'] = tsne[:,1]
    df_subset['y'] = y

    ax = sns.scatterplot(
        x="one", y="two",
        hue="y",
        palette=sns.color_palette("hls", len(np.unique(y)) ),
        data=df_subset,
        legend="full" #, alpha=0.3,  ax=ax3
    )
    plt.show()


def visualize_pair_distributions(x_train, y_train, dataset_name, model_file, x_train_2, y_train_2, dataset_name_2, model_file_2):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df_subset = {}
    number_of_classes = 10 # for visualization readability purposes it is limited to 10
    number_of_instances = 20000 # number of points to be ploted 

    #indices = np.unique(y, return_index=True)[1]
    indices_x = np.where(y < number_of_classes)[0]
    y = np.where(y < number_of_classes)[1]
    #y = y[:number_of_instances]

    # loading model and extracting the image features
    model = load_model(model_file)
    data = X[indices_x]
    #data = data[:number_of_instances]
    
    #print(np.shape(data[:number_of_instances]))
    weights = util.act_func(model, data)

    #between 32 and 64 dim is enough for keep more than 90% of explained variance of the images:
    #https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_On_the_Intrinsic_Dimensionality_of_Image_Representations_CVPR_2019_paper.pdf
    isomap = Isomap(n_components = 32) 
    compressed_data = isomap.fit_transform(weights)

    #tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne = TSNE(n_components=2).fit_transform(compressed_data)
    #print(tsne[:,0])
    #print(tsne[:,1])
    df_subset['one'] = tsne[:,0]
    df_subset['two'] = tsne[:,1]
    df_subset['y'] = y

    ax = sns.scatterplot(
        x="one", y="two",
        hue="y",
        palette=sns.color_palette("hls", len(np.unique(y)) ),
        data=df_subset,
        legend="full" #, alpha=0.3,  ax=ax3
    )
    plt.show()


def plotDistributionByClass(instances, indexesByClass):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)

    for c, indexes in indexesByClass.items():
        X = instances[indexes]
        #reducing to 2-dimensional data
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotAccuracy(arr, steps, label):
    arr = np.array(arr)
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arr, 'k')
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps+1, 10))
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)

    for k, v in distributions.items():
        points = distributions[k]

        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plot(X, y, coreX, coreY, t):
    classes = list(set(y))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color]))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plot2(X, y, t, classes):
    X = classifiers.pca(X, 2)
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        #labels
        classLabels.append('Class {}'.format(cl))

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def finalEvaluation(arrAcc, steps, label):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, steps, label)


def plotF1(arrF1, steps, label):
    arrF1 = np.array(arrF1)
    c = range(len(arrF1))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arrF1, 'k')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if steps > 10:
        plt.xticks(range(1, steps+1, 10))
    else:
        plt.xticks(range(1, steps+1))
    plt.title(label)
    plt.ylabel("F1")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBoxplot(mode, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=90)

    if mode == 'acc':
        plt.title("Accuracy - Boxplot")
        #plt.xlabel('step (s)')
        plt.ylabel('Accuracy')
    elif mode == 'mcc':
        plt.title('Mathews Correlation Coefficient - Boxplot')
        plt.ylabel("Mathews Correlation Coefficient")
    elif mode == 'f1':
        plt.title('F1 - Boxplot')
        plt.ylabel("F1")

    plt.show()


def plotAccuracyCurves(listOfAccuracies, listOfMethods):
    limit = len(listOfAccuracies[0])+1

    for acc in listOfAccuracies:
        acc = np.array(acc)
        c = range(len(acc))
        ax = plt.axes()
        ax.plot(c, acc)

    plt.title("Accuracy curve")
    plt.legend(listOfMethods)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.xticks(range(0, limit, 10))
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBars(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], label=listOfMethods[l], align='center')

    plt.title("Execution time to perform all stream")
    plt.legend(listOfMethods)
    plt.xlabel("Methods")
    plt.ylabel("Execution time")
    plt.xticks(range(len(listOfTimes)))
    plt.show()


def plotBars2(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l])

    plt.title("Average Accuracy")
    plt.xlabel("Methods")
    plt.ylabel("Accuracy")
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfTimes)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plotBars3(listOfAccuracies, listOfMethods):
    
    for l in range(len(listOfAccuracies)):    
        ax = plt.axes()
        ax.bar(l, 100-listOfAccuracies[l])

    plt.title("Average Error")
    plt.xlabel("Methods")
    plt.ylabel("Error")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfAccuracies)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plotBars4(baseline, listOfAccuracies, listOfMethods):
    
    for l in range(1,len(listOfAccuracies)):    
        ax = plt.axes()
        #ax.bar(l, (listOfAccuracies[l]-baseline)/listOfAccuracies[l])
        ax.bar(l, ((listOfAccuracies[l]-baseline)/baseline)*100)
        print('Error reduction:',((listOfAccuracies[l]-baseline)/baseline)*100)

    plt.title("Reduction Percentage Error")
    plt.xlabel("Methods")
    plt.ylabel("% Error under baseline (Static SSL)")
    #plt.yticks(range(0, 101, 10))
    plt.xticks(range(1, len(listOfAccuracies)), listOfMethods[1:])
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def plot_images(title, data, labels, similarities, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        try:
            ax = axes[i//num_col, i%num_col]
            ax.imshow(np.squeeze(data[i]), cmap='gray')
            ax.set_title('{}-Sim={}'.format(labels[i], similarities[i]))
            ax.set_axis_off()
        except Exception as e:
            pass    
    fig.suptitle(title)    
    plt.tight_layout(pad=3.0)
    plt.show()


def run_act_func_animation(model, dataset_name, instances, labels, first_nth_classes, layerIndex, steps, file):
    
    fig = plt.figure()

    def plot_animation(i):
        plt.clf()
        uniform_data = []
        
        for c in range(first_nth_classes):
            ind_class = np.where(labels == c)
            image = np.asarray([instances[ind_class][i]])
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            #print(np.shape(arrWeights))
            uniform_data.append(arrWeights[:first_nth_classes])
        
        ax = sns.heatmap(uniform_data)#, annot=True
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Classes')
        title = 'Activation function of instance {} on GTSRB'.format(i, dataset_name)
        plt.title(title)
        
    animator = ani.FuncAnimation(fig, plot_animation, frames=200, interval = steps)
    animator.save(file, fps=2)
    #plt.show()


### Helper function for run_boxes_analysis()
def print_points_boxes(ax, c, boxes, arr_points, arr_pred, tau=0.0001, dim_reduc_obj=None):
    color={0:'yellow', 1:'green', 2:'blue'}
    arr_polygons = []

    for box in boxes:
        #print(class_to_monitor, box)
        x1 = box[0][0]
        x2 = box[0][1]
        y1 = box[1][0]
        y2 = box[1][1]

        x1 = x1*tau-x1 if x1 > 0 else x1-tau
        x2 = x2*tau+x2 if x2 > 0 else x2+tau
        y1 = y1*tau-y1 if y1 > 0 else y1-tau
        y2 = y2*tau+y2 if y2 > 0 else y2+tau

        rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        #print(rectangle)
        polygon = Polygon(rectangle)
        arr_polygons.append(polygon)

        ax.add_patch(mpatches.Polygon(rectangle, alpha=0.2, color=color[c]))

    for ypred, intermediateValues in zip(arr_pred, arr_points):
        x,y = None, None
        data = np.asarray(intermediateValues)
        #print(np.shape(data))
        
        if dim_reduc_obj!=None:
            dim_reduc_obj = pickle.load(open(dim_reduc_obj, "rb"))
            data = dim_reduc_obj.transform(data.reshape(1, -1))[0] #last version
            #print(np.shape(data))
            x = data[0]
            y = data[1]
        else:
            x = data[0]
            y = data[-1]

        point = Point(x, y)
        is_outside_of_box = True

        for polygon in arr_polygons:
            if polygon.contains(point):
                is_outside_of_box = False
        
        if is_outside_of_box:
            if c != ypred:
                #true positive
                plt.plot([x], [y], marker='.', markersize=10, color="red")
            else:
                #false positive
                plt.plot([x], [y], marker='x', markersize=10, color="red")
        else:
            if c == ypred:
                #true negative
                plt.plot([x], [y], marker='.', markersize=10, color=color[c]) 
            else:
                #false negative
                plt.plot([x], [y], marker='x', markersize=10, color=color[c])


### Function that performs analysis of positives and negatives detections from the OOB-based safety monitors
def run_boxes_analysis(model, dataset_name, technique, instances, labels,\
 first_nth_classes, layerIndex, steps, file, monitor_folder, dim_reduc_obj):
    num_instances = 50
    tau = 0.001 # enlarging factor for abstraction boxes area

    fig, ax = plt.subplots()

    for c in range(first_nth_classes):
        ind_class = np.where(labels == c)
        arr_points = []
        arr_pred = []
        #dim_reduc_obj = os.path.join(monitor_folder, dim_reduc_obj)

        for i in range(num_instances):
            image = np.asarray([instances[ind_class][i]])
            y_pred = np.argmax(model.predict(image))
            arr_pred.append(y_pred)

            boxes_path = os.path.join(monitor_folder, 'class_{}'.format(c), 'monitor_{}_3_clusters.p'.format(technique))
            boxes = pickle.load(open(boxes_path, "rb"))
            #print('boxes shape for class', c, np.shape(boxes))
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arr_points.append(arrWeights)

        print_points_boxes(ax, c, boxes, arr_points, arr_pred, tau, dim_reduc_obj)

    plt.show()


### Helper function for plot_single_clf_pca_actFunc_based_analysis()
def startAnimation(X, y, yt, clf):
    X = np.array(X)
    y = np.array(y)
    yt = np.array(yt)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
        
    # initialization function: plot the background of each frame
    def init():
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,
        
    # animation function.  This is called sequentially
    def animate(i):
        #print('X', np.shape(X))
        #print('y', np.shape(y))
        #print('yt', np.shape(yt))
        
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Class {}".format(i+1))
        #plt.show()
        return scatter,
    
    anim = ani.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=100, blit=True)
    anim.save('animation.mp4', fps=1)
    #plt.show()


### Function that analyzes single classifiers trained on 2D projections from activation funcs
def plot_single_clf_pca_actFunc_based_analysis(model, dataset_name, clf, instances, labels,\
 first_nth_classes, layerIndex, steps, file, dim_reduc_obj):
    
    #fig, ax = plt.subplots()
    #fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8)) 
    '''
    for c in range(first_nth_classes):
        ind_class = np.where(labels == c)
        arr_pred_CNN = []
        arr_pred_monitor = []
        data = []

        for i in range(num_instances):
            image = np.asarray([instances[ind_class][i]])
            y_pred = np.argmax(model.predict(image))
            arr_pred_CNN.append(y_pred)
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arrWeights = np.array(arrWeights).reshape(1, -1)
            
            reduced_data = dim_reduc_obj.transform(arrWeights)
            data.append(reduced_data[0])
            m_pred = clf.predict(reduced_data)
            arr_pred_monitor.append(m_pred[0])
    '''
        #startAnimation(data, arr_pred_CNN, arr_pred_monitor, clf)
    #X = np.array(data)
    #y = np.array(arr_pred_CNN)
    #yt = np.array(arr_pred_monitor)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
        
    # initialization function: plot the background of each frame
    def init():
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,

    def animate(i):
        num_instances = 10
        ind_class = np.where(labels == i)
        arr_pred_CNN = []
        arr_pred_monitor = []
        data = []

        for n in range(num_instances):
            image = np.asarray([instances[ind_class][n]])
            y_pred = np.argmax(model.predict(image))
            arr_pred_CNN.append(y_pred)
            
            arrWeights = util.get_activ_func(model, image, layerIndex=layerIndex)[0]
            arrWeights = np.array(arrWeights).reshape(1, -1)
            
            reduced_data = dim_reduc_obj.transform(arrWeights)
            data.append(reduced_data[0])
            m_pred = clf.predict(reduced_data)
            arr_pred_monitor.append(m_pred[0])
        
        X = np.array(data)
        y = np.array(arr_pred_CNN)
        yt = np.array(arr_pred_monitor)
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Class {}".format(i))
        #plt.show()
        return scatter,
    
    #for i in range(3):
    #    animate(i)
    
    anim = ani.FuncAnimation(fig, animate, init_func=init, frames=30, interval=first_nth_classes, blit=True)
    anim.save(file, fps=1)

    #plt.show()


def visualize_experiments(experiments, threat, names, title, classes_to_monitor):

    arr_readouts = []
    img_folder_path = os.path.join('plots', threat, 'img') 

    for experiment, name in zip(experiments, names):
        avg_cf = {}

        logs = experiment.get_logs()
        #print(logs['True Positive - Class 0'].y) 

        # storing results
        readout = Readout()
        readout.name = name
        
        readout.avg_acc = logs['Accuracy'].y
        readout.avg_time = logs['Process time'].y
        readout.avg_memory = logs['Memory'].y
        readout.avg_F1 = logs['F1'].y

        for class_to_monitor in range(classes_to_monitor):
            fp = 'False Positive - Class {}'.format(class_to_monitor)
            fn = 'False Negative - Class {}'.format(class_to_monitor)
            tp = 'True Positive - Class {}'.format(class_to_monitor)
            tn = 'True Negative - Class {}'.format(class_to_monitor)

            avg_cf.update({class_to_monitor: [int(float(logs[fp].y)), int(float(logs[fn].y)), int(float(logs[tp].y)), int(float(logs[tn].y))]})
        readout.avg_cf = avg_cf

        arr_readouts.append(readout)

    fig_name = img_folder_path+'all_methods_class_'+title+'.pdf'
    os.makedirs(img_folder_path, exist_ok=True)
    metrics.plot_pos_neg_rate_stacked_bars_total(title, arr_readouts, classes_to_monitor, fig_name)


def pos_neg_stacked_bars(title, methods, arr_pos_neg, fig_path):
    figures = []

    y_tn = arr_pos_neg[0]
    y_fp = arr_pos_neg[1] 
    y_fn = arr_pos_neg[2] 
    y_tp = arr_pos_neg[3] 

    #COLOR = 'black'
    #mpl.rcParams['text.color'] = 'white'
    #mpl.rcParams['axes.labelcolor'] = 'black'
    #mpl.rcParams['xtick.color'] = 'black'
    #mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['font.size'] = 12

    xticks = [i for i in range(len(methods))]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.3
    blue = [0, .4, .6]
    yellow = [1, 0.65, 0.25]
    red = [1, 0, 0]
    #darkgrey = 'darkgrey'
    #gray = 'gray'
    #grey = 'grey'
    ax.bar(methods, y_tp, color=blue, edgecolor="white", width=width, label='True positive')
    sums = y_tp
    ax.bar(methods, y_fn, bottom=sums, color=yellow, edgecolor="white", hatch="x", width=width, label='False negative')
    sums =[_x + _y for _x, _y in zip(sums, y_fn)]
    ax.bar(methods, y_fp, bottom=sums, color=red, edgecolor='white', hatch=".", width=width, label='False positive')
    sums = [_x + _y for _x, _y in zip(sums, y_fp)]
    #ax.bar(methods, y_tn, bottom=sums, color=[0, 0.2, 0.1], edgecolor='white', hatch="*", width=width, label='True negative')

    ax.set_xlabel("Methods")
    ax.set_ylabel("Instances")
    #ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks, methods)
    ax.legend()
    #ax.annotate('{}'.format(height))

    for i in range(len(y_fp)):
        plt.annotate(str(y_tp[i]), xy=(width/2+i-0.2, y_tp[i]*0.2), va='bottom', ha='left')
        plt.annotate(str(y_fn[i]), xy=(width/2+i-0.2, (y_fn[i]+y_tp[i])-y_fn[i]*0.5), va='bottom', ha='left')
        plt.annotate(str(y_fp[i]), xy=(width/2+i-0.2, (y_fp[i]+y_fn[i]+y_tp[i])-y_fp[i]*0.5), va='bottom', ha='left')
        
    

    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    figures.append(fig)
    plt.show()

    multipage(fig_path, figures, dpi=250)


def plot_critical_difference(names, avranks, num_datasets):
    import Orange
    import matplotlib.pyplot as plt 
    
    cd = Orange.evaluation.compute_CD(avranks, num_datasets)
    Orange.evaluation.graph_ranks(avranks, names, reverse=False, cd=cd, width=5, textspace=1)
    plt.show()