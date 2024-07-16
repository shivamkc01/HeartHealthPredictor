#!/bin/sh
set -e
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "TRAINING STARTED!"
log "Author: Shivam Chhetry"

# echo "LOGISTIC REGRESSION TRAINING STARTED!"
# python model.py --fold 10 --model lr --logs logisticRegression_model --scale True --metric roc_auc --plot_roc True

# echo "SUCCESSFUL DONE!"s

# echo "================================================================================================================"

# echo "DECISION TREE TRAINING STARTED!"
# python model.py --fold 10 --model dt --logs decisionTree_model --scale True --metric roc_auc --plot_roc True

# echo "SUCCESSFUL DONE!"s

# echo "================================================================================================================"


# echo "NAIVE BAYES TRAINING STARTED!"
# python model.py --fold 10 --model nb --logs naiveBayes_model --scale True --metric roc_auc --plot_roc True

# echo "SUCCESSFUL DONE!"s


# echo "================================================================================================================"


# echo "SVM TRAINING STARTED!"
# python model.py --fold 10 --model svm --logs svm_model --scale True --metric roc_auc --plot_roc True

# echo "SUCCESSFUL DONE!"s



echo "KMeans Cluster TRAINING STARTED!"
python model.py --fold 10 --model kmc --logs kmc_model --scale True --metric roc_auc --plot_roc True

echo "SUCCESSFUL DONE!"s