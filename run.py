import sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import os

################################
# Complete the functions below #
class SoftGaussianParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        weights = np.zeros((num_test, self.n_classes)) #put zeros matrix there, check if its okay
        classes_pred = np.zeros(num_test)

        for (index,row) in enumerate(test_data):
            eucl_dist_matrix = np.sqrt(np.sum((self.train_inputs - row)**2.0, axis=1))
            normalizing_factor=1/(num_test*np.sqrt(2*self.sigma*np.pi))
            Gaus_kernel_dist = normalizing_factor * np.exp((-1/(2*(self.sigma**2))) * (eucl_dist_matrix**2))

            for (i,values) in enumerate(Gaus_kernel_dist):
                weights[index,int(self.train_labels[i])-1]+= values

            classes_pred[index] = np.argmax(weights[index,:])+1 #because values are from 1 to 3 but recheck
        return classes_pred




class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def soft_parzen(self, sigma):
        f= SoftGaussianParzen(sigma)
        f.train(self.x_train,self.y_train)
        preds = f.compute_predictions(self.x_val)
        return np.sum(np.abs(preds-self.y_val))/len(preds)


def get_test_errors(thyroid,sigma):
    (p1,p2,p3,test)= split_fold(thyroid)
    train_set=np.concatenate((p1,p2,p3), axis=0)
    f= SoftGaussianParzen(sigma)
    f.train(train_set[:,1:],train_set[:,0])
    preds= f.compute_predictions(test[:,1:])
    confusion_matrix= np.zeros((3,3))
    labels= test[:,0]

    for i in range(len(preds)):
        confusion_matrix[int(preds[i])-1][int(labels[i])-1]+=1

    preds=['Predicted normal','Predicted hyperthyroid','Predicted hypothyroid']
    act=['Actual normal','Actual hyperthyroid','Actual hypothyroid']
    fig=plt.figure()
    ax=fig.add_subplot(111)
    c=ax.matshow(confusion_matrix/confusion_matrix.max(), cmap='gray')
    fig.colorbar(c)
    ax.set_xticklabels(['']+act)
    ax.set_yticklabels(['']+preds)
    plt.show()
    #return confusion_matrix

def split_fold(thyroid):
    parts = thyroid.shape[0]//4
    return(thyroid[0:parts,:], thyroid[parts:parts*2,:],thyroid[parts*2:parts*3,:], thyroid[parts*3:,:])


def kfold_cross_validation(thyroid):
    np.random.shuffle(thyroid)
    (p1,p2,p3,test)= split_fold(thyroid)
    train_1=np.concatenate((p2,p3), axis=0)
    val_1= p1
    cross_1=ErrorRate(train_1[:,1:],train_1[:,0],val_1[:,1:],val_1[:,0])

    train_2=np.concatenate((p3,p1), axis=0)
    val_2= p2
    cross_2=ErrorRate(train_2[:,1:],train_2[:,0],val_2[:,1:],val_2[:,0])

    train_3=np.concatenate((p2,p1), axis=0)
    val_3= p3
    cross_3=ErrorRate(train_3[:,1:],train_3[:,0],val_3[:,1:],val_3[:,0])

    test_parameter= [0.01,0.02,0.08,0.1,0.25,0.50,1.0,1.25,1.50,1.75,2.0,3.0,5.0]
    sigma_err=[]
    for p in test_parameter:
        sigma_err += [cross_1.soft_parzen(p) + cross_2.soft_parzen(p) + cross_3.soft_parzen(p)]


    return test_parameter[np.argmin(sigma_err)]
################################

# Download/create the dataset
def fetch():
    h= urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data', 'new-thyroid2.txt') # replace this with code to fetch the dataset

# Train your model on the dataset
def train():
    thyroid= np.genfromtxt('new-thyroid2.txt', delimiter=',')
    np.random.seed(30)
    np.random.shuffle(thyroid)
    np.savetxt('sigmatrain.txt', [kfold_cross_validation(thyroid)])

# Compute the evaluation metrics and figures
def evaluate():
    thyroid= np.genfromtxt('new-thyroid2.txt', delimiter=',')
    np.random.seed(30)
    np.random.shuffle(thyroid)
    sigma=np.genfromtxt('sigmatrain.txt', delimiter=',')
    get_test_errors(thyroid, sigma) # replace this with code to evaluate what must be evaluated

# Compile the PDF documents
def build_paper():
    os.system("card.pdf")
    os.system("paper.pdf")

###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
    print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
    sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
    supported_functions[arg]()
else:
    raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))