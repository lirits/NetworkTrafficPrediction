import matplotlib.pyplot as plt
from tools import preprocess_prediction


def plot_batch_prediction(
        net,
        dataloader,
        device: str = 'cpu',
        start_time: int = 0,
        time_step: int = 24 * 7,
        fig_size: tuple = (
            20,
            8),
        dpi: int = 200):
    Y, Y_pred = preprocess_prediction(net, dataloader, device)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.plot(Y[start_time:time_step], label='True Traffic')
    plt.plot(Y_pred[start_time:time_step], label='Predict Traffic')
    plt.grid()
    plt.legend()
    plt.title('Network Traffic Prediction')
    plt.xlabel('Time')
    plt.ylabel('Traffic (Fixed)')
    plt.show()
