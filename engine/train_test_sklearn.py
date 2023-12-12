from sklearn import *
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier


def run_train_test_sklearn(train_dataset, valid_dataset, test_dataset,logging, **params):
    # read data and labels
    X_train, y_train = train_dataset.get_embeddings(), train_dataset.get_labels() 
    X_valid, y_valid = valid_dataset.get_embeddings(), valid_dataset.get_labels()
    X_test, y_test = test_dataset.get_embeddings(), test_dataset.get_labels() 
    # choose between small train dataset or all train dataset\
    if params['dataset'] == 'small':
        X_train, y_train = X_train[-10000:], y_train[-10000:]
    # choose between methods
    method = params['method']
    # assert method in ['svm']
    if method == 'svm':
        if params['svm_gamma'] != 'auto' and params['svm_gamma'] != 'scale':
            gamma = float(params['svm_gamma'])
        else:
            gamma = params['svm_gamma']
        model = sklearn.svm.SVC(kernel=params['svm_kernel'], C=float(params['svm_C']), gamma=gamma, class_weight='balanced')
        
        dict={'method':method,'dataset': params['dataset'],'kernel':params['svm_kernel'],'C':float(params['svm_C']),'gamma':gamma,'class_weight':'balanced'}
        
    elif method == 'decision_tree':
        
        max_depth = params['dt_max_depth']
        min_samples_split = params['dt_min_samples_split']
        min_samples_leaf = params['dt_min_samples_leaf']

        if max_depth is not None and max_depth != 'None':
            max_depth = int(max_depth)

        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)

        model = DecisionTreeClassifier(max_depth=max_depth, 
                                    min_samples_split=min_samples_split, 
                                    min_samples_leaf=min_samples_leaf)
        
        dict={'method':method,'dataset': params['dataset'],'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}
        
    elif method == 'logistic_regression':
        
        C = float(params['lr_C'])  
        solver = params['lr_solver']  ## 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
        max_iter = int(params['lr_max_iter'])  
    
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        
        dict={'method':method,'dataset': params['dataset'],'C':C,'solver':solver,'max_iter':max_iter}
        
    elif method == 'naive_bayes':
        
        var_smoothing = float(params['nb_var_smoothing']) ### 1e-9))

        model = GaussianNB(var_smoothing=var_smoothing)
        
        dict={'method':method,'dataset': params['dataset'],'priors':params['priors'],'var_smoothing':var_smoothing}
        
    elif method == 'random_forest':
        
        n_estimators = int(params['rf_n_estimators']) ##  100  # number of trees
        max_depth = params['rf_max_depth']  # maximum depth of trees
        min_samples_split = int(params['rf_min_samples_split'])  ## 2  # minimum sample split
        min_samples_leaf = int(params['rf_min_samples_leaf'])  # minimum sample leafs

        if max_depth is not None and max_depth != 'None':
            max_depth = int(max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split, 
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight='balanced')
        
        dict={'method':method,'dataset': params['dataset'],'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'class_weight':'balanced'}
        
    elif method == 'adaboost':
        n_estimators = int(params['ab_n_estimators'])  # the number of weak learners
        learning_rate = float(params['ab_learning_rate'])  # learning rate

        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        
        dict={'method':method,'dataset': params['dataset'],'n_estimators':n_estimators,'learning_rate':learning_rate}
    
    elif method == 'mlp':
    
        hidden_layer1_sizes = int(params['mlp_hidden_layer1_sizes'])
        hidden_layer2_sizes = int(params['mlp_hidden_layer2_sizes'])
        if hidden_layer2_sizes < 1:
            hidden_layer_sizes = (hidden_layer1_sizes)
        else:
            hidden_layer_sizes = (hidden_layer1_sizes, hidden_layer2_sizes)

        # print(hidden_layer_sizes)
        activation = params['mlp_activation']  # 'identity', 'logistic', 'tanh', 'relu'
        solver = params['mlp_solver']  # 'lbfgs', 'sgd', 'adam'
        max_iter = int(params['mlp_max_iter'])
        learning_rate_init = float(params['mlp_learning_rate_init'])

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                            solver=solver, max_iter=max_iter, learning_rate_init=learning_rate_init)
        
        dict = {'method': method, 'dataset': params['dataset'], 'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation, 'solver': solver, 'max_iter': max_iter, 
                'learning_rate_init': learning_rate_init}

    
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    valid_accuracy = sklearn.metrics.accuracy_score(y_valid, y_valid_pred)
    valid_roc_auc = sklearn.metrics.roc_auc_score(y_valid, y_valid_pred)
    valid_f1 = sklearn.metrics.f1_score(y_valid, y_valid_pred, average='weighted')
    print(dict)
    print(f'Validation set accuracy:{valid_accuracy}, roc:{valid_roc_auc}, f1 score:{valid_f1}')
    print(sklearn.metrics.classification_report(y_valid, y_valid_pred))
    logging.info(dict)
    logging.info(f'Validation set accuracy:{valid_accuracy}, roc:{valid_roc_auc}, f1 score:{valid_f1}')
    logging.info(sklearn.metrics.classification_report(y_valid, y_valid_pred))
    y_test_pred = model.predict(X_test)
    test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
    test_roc_auc = sklearn.metrics.roc_auc_score(y_test, y_test_pred)
    test_f1 = sklearn.metrics.f1_score(y_test, y_test_pred, average='weighted')
    print(f'Test set accuracy:{test_accuracy}, roc:{test_roc_auc}, f1 score:{test_f1}')
    print(sklearn.metrics.classification_report(y_test, y_test_pred)) 
    logging.info(f'Test set accuracy:{test_accuracy}, roc:{test_roc_auc}, f1 score:{test_f1}')
    logging.info(sklearn.metrics.classification_report(y_test, y_test_pred))
        