# Laboratory practice 2.2: KNN classification
import seaborn as sns
sns.set_theme()
import numpy as np  
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')



def minkowski_distance(a:list , b:list, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """

    # TODO

    sumatory = 0 
    for i in range (len(a)): 
        sumatory += float(abs (a[i] - b[i]) ** p)
        
    return sumatory** (1/p)


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        # simplemente declaramos los datos dados como atributos del objeto 
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        # primero, chequeamos que los parámetros son correctos 
        
        if X_train.shape[0] == y_train.shape[0]: 
            if type(k) == int and k > 0  : 
                if  type(p) == int and p> 0  : 
                    # todos los parámetros han sido chequeados hasta aquí 
                    self.k = k 
                    self.p = p
                    self.x_train = X_train
                    self.y_train = y_train
                                
                else: 
                    raise ValueError("k and p must be positive integers.")
            else: 
                raise ValueError ("k and p must be positive integers.")
        else: 
            raise ValueError("Length of X_train and y_train must be equal.")
        
       
                    
   
    def find_nearest_kneighours (self, x): 
        distances = []

        #comparamos este dato con todas las distancias de nuestro train set 

        for i in range (len(self.x_train)):
            train_data = self.x_train[i] 
            label = self.y_train[i]
            # calculamos la distancia de este dato nuevo con todos los datos que tenemos 
            distance = minkowski_distance(x, train_data, p = self.p)

            # añadimos esta distancia y su correspondiente etiqueta a una lista de distancias 
            distances.append((distance, label))
            
        # ordenamos la lista de distancias por la distancia en orden ascendente 
        ordered_distances = sorted(distances, key=lambda x : x[0])
        # de todos los nodos ordenados, cojo los k primeros 
        nearest = ordered_distances[:self.k]
        return nearest 
        

    def predict(self, X: np.ndarray) -> np.ndarray:
       
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predicted_classes = []

        # para cada dato de la lista X: 
        for x in X:
            # calculo los nodos más cercanos a cada dato 
            nearest = self.find_nearest_kneighours(x)

            # de los nodos más cercanos, cojo sus etiquetas y cuento la frecuencia con la que aparecen 
            labels = {}
            for  (_, label) in nearest : 
                 # coge la frecuencia de las etiquetas viendo si hay alguna clase que no aparezca 
                labels[label] = labels.get(label, 0) + 1

            #finalmente, la etiqueta para ese dato de X será la más frecuente entre sus vecinos 
            clase_max = max(labels, key=labels.get) 
            predicted_classes.append(clase_max)

        return np.array(predicted_classes)



    def predict_proba(self, X:np.ndarray)-> np.ndarray:

        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # lista de clases totales posiblles 
        class_probabilities = []

        # para cada dato de la lista X: 
        for x in X:
            # calculo los nodos más cercanos a cada dato 
            nearest = self.find_nearest_kneighours(x)

            # creamos un diccionario donde la clave es la clase y su valor un contador 
            labels = {0: 0, 1: 0}  
            for _, label in nearest:
                labels[label] += 1

            #ahora, en vez de escoger una clase como tal, devuelvo un diccionario con cada clase y la prob de que sea de esa clase 
            # la probabilidad la calculamos cogiendo el contador de labels y dividiéndolo por el numero de vecinos totales (self.k)

            labels[0] /= self.k
            labels[1] /= self.k

            # añadimos a class_probabilities los resultados 
            class_probabilities.append([labels[0], labels[1]])
        
        return np.array(class_probabilities)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distances = []
        #simplmenete recorremos el training set y calculamos la distancia con el punto dado. 
        for x in self.x_train:
            distances.append(minkowski_distance(x, point)) 

        return np.array(distances)


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        sorted_distances = np.argsort(distances)
        nearest = sorted_distances[:self.k]
        #TODO 
        return nearest 


    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        # cogemos una lista de todas las clases que hay , sin repetir
        labels = np.unique(knn_labels)
        #hacemos un contador para cada una de esas clases 
        counts = [ 0 for i in labels]

        for label in knn_labels : 
            # coge el indice en labels y suma +1 a su contador 
            index = labels[label]
            counts[index]+=1
        
        max_count = max(counts)
        index_max = counts.index(max(counts))
        return labels[index_max]


    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)
    print (" datos predecidos")

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors = "black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
  
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true]) #lista de las etiquetas reales
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred]) #lista de las etiquetas estimadas 

    # Confusion Matrix
    # TODO
    tp = 0
    tn = 0 
    fp = 0 
    fn = 0
    # recorremos los indices y vemos si coindicen o no los valores entre ambas listas: 

    for i in range(len(y_true_mapped)): 
        true_label = y_true_mapped [i]
        predicted_label = y_pred_mapped[i]
        
        #el test ha acertado 
        if predicted_label == 1 and true_label == 1:
            tp += 1  
        elif predicted_label == 0 and true_label == 0:
            tn += 1  
        
        #el test ha fallado 
        elif predicted_label == 1 and true_label == 0:
            fp += 1  
        elif predicted_label == 0 and true_label == 1:
            fn += 1 
    # Accuracy
    # TODO
    accuracy  = 0.0
    if tp + tn + fp + fn != 0 : 
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    # TODO
    precision  = 0.0
    if tp + fp != 0 : 
        precision = tp / (tp + fp)

    # Recall (Sensitivity)
    # TODO
    recall = 0.0
    if tp + fn != 0 : 
        recall = tp / (tp + fn)
    # Specificity

    # TODO
    specificity  = 0.0 
    if tn + fp != 0:
        specificity = tn / (tn + fp)

    # F1 Score
    # TODO
    f1 = 0.0
    if precision + recall != 0: 
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    # TODO

    bin_centers = []
    true_proportions = []
    # dividimos en bins por "rangos" de probabilidades
    # si los bins son = 10 , los rangos irán 0-0,1; 0-0,2 ,... 
    particiones = np.linspace(0,1, n_bins +1 )

    for j in range(n_bins): 
       
        #definimos los limites del rango  
        lim_inf = particiones[j]
        lim_sup = particiones[j + 1]

        # guardamos los indices de aquellas probabilidades que se encuentren en este rango 
        y_index = [i for i in range(len(y_probs)) if lim_inf <= y_probs[i] < lim_sup]
        # de esos indices , cogemos las labels correspondientes 
        labels = [y_true[i] for i in y_index]
        if labels : 
            # calculamos cuantas etiquetas positivas hay , mappeando las etiquetas 
            num_positive = sum(1 for lab in labels if lab == positive_label )
            positive_fraction = num_positive / len(labels)

            #añadimos el centro del rango a la lista correspondiente 
            bin_centers.append((lim_sup + lim_inf) / 2)

            # añadimos la fraccion de positivos a la otra lista 
            true_proportions.append(positive_fraction)

    #cuando ya tenemos las dos listas hechas, las ploteamos 
    plt.plot(bin_centers, true_proportions, marker='o', linestyle='-', label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    # TODO
    #mapeamos las etiquetas segun la etiqueta positiva 
    y_true_mapped = []
    y_true_mapped= np.array(y_true_mapped) 
    y_true_mapped = (y_true == positive_label)

    # calculamos las listas de las probabilidades para positivo y negativo 
    predicted_positive = np.array(y_probs)[np.array(y_true_mapped) == 1]
    predicted_negative = np.array(y_probs)[np.array(y_true_mapped) == 0]


    #ploteamos ahora los dos histogramas : 

    plt.hist(predicted_positive, bins=n_bins)
    plt.xlabel("Predicted Probability")
    plt.ylabel('Count')
    plt.title('Positive Class Predictions ')
    plt.show()
    
    plt.hist(predicted_negative, bins=n_bins)
    plt.xlabel("Predicted Probability")
    plt.ylabel('Count')
    plt.title('Negative Class Predictions ')
    plt.show()



    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    # TODO
    y_true_mapped = [1 if i == positive_label else 0 for i in y_true]

    fpr = [] # fpr = fp / (fp + tn)
    tpr = [] # tpr = tp / (tp + fn )
    thresholds = np.arange(0, 1.1, 0.1)
    for j in thresholds: 
        # las y_predecidas serán las que sean mayores que el threshold 
        y_pred_mapped =  [1 if prob >= j else 0 for prob in y_probs]

        tp = 0
        tn = 0 
        fp = 0 
        fn = 0
        for i in range(len(y_true_mapped)): 

            true_label = y_true_mapped [i]
            predicted_label = y_pred_mapped[i]
    
            if true_label == predicted_label: 
                # el test ha acertado -> se añade a la lista correspondiente 
                if true_label == 1:  # Porque ya lo convertiste en 0 o 1
                    tp +=1 
                else: 
                    tn +=1 
            else: #el test ha fallado -> se convierte en falso 
                # si habia puesto que era cierta, es un falso positivo 
                if predicted_label == 1 : 
                    fp +=1 
                else: 
                    # si había puesto que era negativa, entonces es falso negativo 
                    fn +=1 




        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)


    plt.plot(fpr, tpr, marker='o', linestyle='-', color='b', label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Model")  # Línea diagonal aleatoria
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}


  
