

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    count=0
    y_hat=y_hat.tolist()
    y=y.tolist()
    for i in range(len(y_hat)):
        if y_hat[i]==y[i]:
            count=count+1
    
    acc=float(count)/float(len(y_hat))
    return acc

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y_hat=y_hat.tolist()
    y=y.tolist()
    count=0
    total=0
    for i in range(len(y_hat)):
        if y_hat[i] == cls and y_hat[i] == y[i]:
            count+=1
            total+=1
        elif y_hat[i] == cls:
            total+=1
    if total==0:
        return 0
    acc=float(count)/float(total)
    return acc



def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y_hat=y_hat.tolist()
    y=y.tolist()
    count=0
    total=0
    for i in range(len(y_hat)):
        if y[i] == cls and y_hat[i] == y[i]:
            count+=1
            total+=1
        elif y[i] == cls:
            total+=1
    if total==0:
        return 0
    acc=float(count)/float(total)
    return acc

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y_hat=y_hat.tolist()
    y=y.tolist()
    acc=0
    for i in range(len(y_hat)):
        acc=acc+((y_hat[i]-y[i])**2)
    acc=float(acc)/float(len(y_hat))
    acc=(acc)**(1/2)

    return acc


def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y_hat=y_hat.tolist()
    y=y.tolist()
    acc=0
    for i in range(len(y_hat)):
        acc=acc+(abs(y_hat[i]-y[i]))
    acc=float(acc)/float(len(y_hat))

    return acc
