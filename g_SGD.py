from typing import Tuple
import torch
import numpy as np
import pandas as pd
torch.set_default_dtype(torch.float)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
df = df.sample(frac=1)#shuffle

#add label indices column
mapping = {k:v for v,k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)

#normalize data
alldata = torch.tensor(df.iloc[:,[0,1,2,3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

#create datasets
targets_tr = torch.tensor(df.iloc[:100,5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[100:,5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]


# This function is used to define the structure of the MLP model
def MLP (input_size, hidden_size, num_classes):
    # set the same seed for generating same initial weight matrix and bias vector
    torch.manual_seed(3)
    W1 = torch.randn(input_size,hidden_size,requires_grad=True)
    W2 = torch.randn(hidden_size,hidden_size,requires_grad=True)
    W3 = torch.randn(hidden_size,num_classes,requires_grad=True)
    B1 = torch.randn(hidden_size, requires_grad=True)
    B2 = torch.randn(hidden_size, requires_grad=True)
    B3 = torch.randn(num_classes, requires_grad=True)
    return W1,W2,W3,B1,B2,B3

# This function is used to construct the Skeleton Weights Flag matrix
# The input is the Weight matrix
def getSkeletonWeightsFlag(fcweight, layer):
    # get the size of the input weight matrix
    row, col = np.shape(fcweight)
    # init the flag matrix
    fc_skeletonWeightsFlag = np.zeros((row, col))

    # using the method mentioned following section of the paper
    # 4.1 SKELETON METHOD
    # 1. Construct skeleton weights
    if (row == col): # for all hidden layer weight matrix
        for i in range(col):
            fc_skeletonWeightsFlag[i][i] = 1

    elif (layer == 1): # for the first layer weight matrix
        for i in range(col):
            fc_skeletonWeightsFlag[i % row][i] = 1

    elif (layer == MLP_LAYERS):# for the last layer weight matrix
        for i in range(row):
            fc_skeletonWeightsFlag[i][i % col] = 1
    return fc_skeletonWeightsFlag

# This function is used to find the weights contained in a basis path and compute its path value
# The inputs are Weight matrix and Skeleton Weights Flag matrix
def getBasisPathValue(fcweight1, fcweight2, fcweight3, skweightFlag1, skweightFlag2, skweightFlag3):
    # initial the basis path value vector
    Basic_path_value = torch.zeros(m - H)
    # initial the basis path weight index matrix
    # which giving the index vector (size is 1*MLP_LAYERS) of the weights in the Weight Vector that contained in every basis path
    Basic_path_weights_index = torch.zeros(m - H, MLP_LAYERS)
    index = 0
    # search every path from the input nodes
    # i,h,j,k refer the index of the node in each node layer respectively
    for i in range(INPUT_SIZE):
        for h in range(HIDDEN_SIZE):
            for j in range(HIDDEN_SIZE):
                for k in range(NUM_CLASSES):
                    # if the sum of the flag >=2, the path implied in the node index vector is a basis path
                    if (skweightFlag1[i][h] + skweightFlag2[h][j] + skweightFlag3[j][k] >= 2):
                        # conpute the path value by multiply the weights contained in the path
                        Basic_path_value[index] = (fcweight1[i][h] * fcweight2[h][j] * fcweight3[j][k]).clone()

                        # find the index of the basis path weight in the Weight Vector
                        Basic_path_weights_index[index] = torch.tensor(
                            [i * HIDDEN_SIZE + h, h * HIDDEN_SIZE + j + Num_W1, j * NUM_CLASSES + k + Num_W1 + Num_W2])
                        index += 1
    return Basic_path_value, Basic_path_weights_index

# This function is used to construct the Matrix G and Matrix A'
# The inputs are the outputs from function getBasisPathValue()
def composeMatrix_GandMatrix_Apr(basic_path_value,basic_path_weights_index,w_vector):
    # initi the size of these two matrix
    matrix_g = torch.zeros(m-H,m)
    matrix_apr = torch.zeros(m-H,m)
    # construct the matrix by rows
    for i in range(m-H):
        for j in range(m):
            # if the weights index is found in basis path value index vector which means the weight is in a basis path
            if(j in basic_path_weights_index[i]):
                # set elements by method mentioned in section 4.2 of the paper
                matrix_g[i][j] = basic_path_value[i]/w_vector[j]
                matrix_apr[i][j] = 1

            # else set elements to 0
            else:
                matrix_g[i][j] = 0
                matrix_apr[i][j] = 0

    return matrix_g, matrix_apr.t() # A' is the transpose of matrix_apr

# This function is written to construct A^ but not used since our MLP model is not applicable
def composeMatrix_Ahat (matrix_apr):
    I = torch.eye(H)
    O = torch.zeros(m-H,H)
    x = torch.cat((I,O),0)
    matrix_ahat = torch.cat((x,matrix_apr),1)
    return matrix_ahat

# This function is used to reshape the Weight matrix and corresponding Gradient matrix to vectors (size is 1*m)
# The inputs are Weights matrix and Gradients matrix
def getWeightGradientVector(fcweight1,fcweight2,fcweight3,dfcweight1,dfcweight2,dfcweight3):

    #reshape Gradient Matrix to Vector
    dw1_vector = torch.reshape(dfcweight1,(-1,))
    dw2_vector = torch.reshape(dfcweight2,(-1,))
    dw3_vector = torch.reshape(dfcweight3,(-1,))
    #connect Gradient Matrixs
    dw_vector = torch.cat((dw1_vector,dw2_vector,dw3_vector),0)
    #reshape Weight Matrix to Vector
    w1_vector = torch.reshape(fcweight1,(-1,))
    w2_vector = torch.reshape(fcweight2,(-1,))
    w3_vector = torch.reshape(fcweight3,(-1,))
    #connect Weight Matrixs
    w_vector = torch.cat((w1_vector,w2_vector,w3_vector),0)
    return dw_vector, w_vector

# This function is used to do "dot operation"
def dotOperation(W, A):
    row_A, col_A = A.size()
    WA = torch.zeros(col_A)
    for j in range (col_A):
        WA[j] = torch.prod(torch.pow(torch.abs(W),A[:,j].t()))
    return WA

# This funtion is used to compute the accuracy of the parameters
def getAccuracy(outpt,label):
    _, predict = torch.max(outpt, 1)
    correctNum = (predict == label).sum()
    acc = 100 * correctNum / len(label)
    return acc

# This function is used to test the parameters on validation set
def dataTest(data,target,weight1,weight2,weight3,bias1,bias2,bias3):
    data_output = torch.relu(torch.relu(data @ weight1 + bias1) @ weight2 + bias2) @ weight3 + bias3
    data_loss = torch.nn.functional.cross_entropy(data_output, target)
    data_acc = getAccuracy(data_output, target)
    return data_loss,data_acc

# define the size of the MLP model
MLP_LAYERS = 3
INPUT_SIZE = 4
HIDDEN_SIZE = 12
NUM_HIDDEN_NODE_LAYER = 2
NUM_CLASSES = 3


# initial the MLP
w1, w2, w3, b1, b2, b3 = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
w1_pr, w2_pr, w3_pr, b1_pr, b2_pr, b3_pr = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
# m is the number of the weights (or edges) in the MLP model
m = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE ** NUM_HIDDEN_NODE_LAYER + HIDDEN_SIZE * NUM_CLASSES
# H is the number of the hidden nodes
H = HIDDEN_SIZE * NUM_HIDDEN_NODE_LAYER
# get flag matrix
w1flag = getSkeletonWeightsFlag(w1, 1)
w2flag = getSkeletonWeightsFlag(w2, 2)
w3flag = getSkeletonWeightsFlag(w3, 3)
# get the number of elements in Weight matrix
Num_W1 = INPUT_SIZE * HIDDEN_SIZE
Num_W2 = HIDDEN_SIZE * HIDDEN_SIZE
Num_W3 = HIDDEN_SIZE * NUM_CLASSES

inputs = data_tr
labels = targets_tr
Num_epochs = 5

Epochs = np.zeros(Num_epochs)
acc_val_pr = np.zeros(Num_epochs)
loss_val_pr = np.zeros(Num_epochs)
acc_val = np.zeros(Num_epochs)
loss_val = np.zeros(Num_epochs)

acc_tr_pr = np.zeros(Num_epochs)
loss_tr_pr = np.zeros(Num_epochs)
acc_tr = np.zeros(Num_epochs)
loss_tr = np.zeros(Num_epochs)

lr = 0.001
Init_Loss, Init_Acc = dataTest(data_tr,targets_tr,w1,w2,w3,b1,b2,b3)
print("Init loss",Init_Loss)
print("Init acc", Init_Acc)

for e in range(Num_epochs):

    #### Normal SGD START

    # set weight gradient and bias gradient to None
    w1_pr.grad = None
    w2_pr.grad = None
    w3_pr.grad = None
    b1_pr.grad = None
    b2_pr.grad = None
    b3_pr.grad = None

    # training the data
    outputs_pr = torch.relu(torch.relu(inputs @ w1_pr + b1_pr) @ w2_pr + b2_pr) @ w3_pr + b3_pr
    # loss function is the cross_entropy
    loss_pr = torch.nn.functional.cross_entropy(outputs_pr, labels)
    # get gradient
    loss_pr.backward()

    # update the weight matrix (normal-SGD)
    w1_pr_temp = w1_pr - lr * w1_pr.grad
    w2_pr_temp = w2_pr - lr * w2_pr.grad
    w3_pr_temp = w3_pr - lr * w3_pr.grad
    w1_pr.data = w1_pr_temp
    w2_pr.data = w2_pr_temp
    w3_pr.data = w3_pr_temp

    # update the bias vector (normal-SGD)
    b1_pr_temp = b1_pr - lr * b1_pr.grad
    b2_pr_temp = b2_pr - lr * b2_pr.grad
    b3_pr_temp = b3_pr - lr * b3_pr.grad
    b1_pr.data = b1_pr_temp
    b2_pr.data = b2_pr_temp
    b3_pr.data = b3_pr_temp

    ##### Normal SGD END




    ##### g-SGD START

    # set weight gradient and bias gradient to None
    w1.grad = None
    w2.grad = None
    w3.grad = None
    b1.grad = None
    b2.grad = None
    b3.grad = None

    # training the data
    outputs = torch.relu(torch.relu(inputs @ w1 + b1) @ w2 + b2) @ w3 + b3
    # loss function is the cross_entropy
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    # get gradient
    loss.backward()

    # get Gradient Matrix
    dw1 = w1.grad
    dw2 = w2.grad
    dw3 = w3.grad

    # reshape the Weight Matrix and Gradient Matrix to vector
    dw, w = getWeightGradientVector(w1, w2, w3, dw1, dw2, dw3)

    # get Path_value Vector and corresponding node index
    Vp, Ip = getBasisPathValue(w1, w2, w3, w1flag, w2flag, w3flag)

    # get Matrix_G and Matrix_Apr ( the maximal linearly independent group )
    matrix_G, matrix_Apr = composeMatrix_GandMatrix_Apr(Vp, Ip, w)

    # using pesudo inverse matrix of G to compute Path_value Gradient Vector
    dvp = dw @ torch.pinverse(matrix_G)

    # sgd on Path_value
    Vp_new = Vp - lr * dvp

    # compute Ratio of Path_value
    R_Vp = Vp_new / Vp

    # compute Ratio of Weights by using  pesudo inverse matrix of A_pr with dotOperation
    r_Wm = dotOperation(R_Vp, matrix_Apr.pinverse())

    # updata the weights by using r_Wm
    w_new = w * r_Wm

    # reshape the new weight vector to matrix
    w1_temp = torch.reshape(w_new[0:Num_W1], (INPUT_SIZE, HIDDEN_SIZE))
    w2_temp = torch.reshape(w_new[Num_W1:Num_W1 + Num_W2], (HIDDEN_SIZE, HIDDEN_SIZE))
    w3_temp = torch.reshape(w_new[Num_W1 + Num_W2:], (HIDDEN_SIZE, NUM_CLASSES))

    # updata the weight matrix (g-SGD)
    w1.data = w1_temp
    w2.data = w2_temp
    w3.data = w3_temp

    # update the bias vector (g-SGD)
    b1_temp = b1 - lr * b1.grad
    b2_temp = b2 - lr * b2.grad
    b3_temp = b3 - lr * b3.grad
    b1.data = b1_temp
    b2.data = b2_temp
    b3.data = b3_temp

    ##### g-SGD END

    # The following codes are written to record the results in every epochs
    # record the loss and acc (Train data) Normal SGD
    loss_tr_pr[e], acc_tr_pr[e] = dataTest(data_tr, targets_tr, w1_pr, w2_pr, w3_pr, b1_pr, b2_pr, b3_pr)
    #record the loss and acc (Train data) g-SGD
    loss_tr[e], acc_tr[e] = dataTest(data_tr,targets_tr,w1,w2,w3,b1,b2,b3)

    #test on the validation data Normal SGD
    loss_val_pr[e], acc_val_pr[e] = dataTest(data_va, targets_va, w1_pr, w2_pr, w3_pr, b1_pr, b2_pr, b3_pr)
    #test on the validation data g-SGD
    loss_val[e], acc_val[e] = dataTest(data_va, targets_va, w1, w2, w3, b1, b2, b3)

# The following codes are written to print the results
print("####\nTrain Result")
print("Train loss in normal space",loss_tr_pr)
print("Train loss in g space",loss_tr)
print("Train acc in normal space",acc_tr_pr)
print("Train acc in g space",acc_tr)
print("####\nEnd")

print("####\nValidation Result")
print("Validation Loss in Normal Space",loss_val_pr)
print("Validation Accuracy in Normal Space",acc_val_pr)
print("Validation Loss in g Space",loss_val)
print("Validation Accuracy in g Space",acc_val)
print("####\nEnd")