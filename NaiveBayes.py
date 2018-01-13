import pickle
from nltk import NaiveBayesClassifier

def features(name):
    features = {}
    features['accumulatedYawRotation'] = name[2]
    features['peakRate'] = name[3]
    features['yawThreshold.bool'] = name[4]
    features['rawThreshold.bool'] = name[5]
# trainclassifier

f1 = open("X_train.txt")
f2 = open("Y_train.txt")

trainer = NaiveBayesClassifier.train
namelist = ([(name, 'X_Train') for name in f1] +
                [(name, 'y_train') for name in f2])

train = namelist[:5000]

classifier = trainer( [(features(n), g) for (n, g) in train] )

with open('classifier.pickle', 'wb') as outfile:
    pickle.dump(classifier, outfile)
    outfile.close()

#use classifier

    try:
        f = open('classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
        classifier.show_most_informative_features(n=10)
        print
        "Classifier result: " + classifier.classify(features("Swapnil"))

    except:
        print
        "Prepare the classifier using - train_classifier.py and then try to use."