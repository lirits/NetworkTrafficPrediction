import matplotlib.pyplot as plt


def plot_raw(y_true,y_pred,label=None,grid=None,figsize=(20,8),dpi=200,mode='train'):
    assert mode in ['train','val','test']
    plt.figure(figsize=figsize,dpi=dpi)
    plt.plot(y_true,label='True')
    if mode == 'test':
        plt.plot(y_pred,label='Prediction')
    if grid:
        plt.grid()
    if label:
        plt.legend()
    plt.show()