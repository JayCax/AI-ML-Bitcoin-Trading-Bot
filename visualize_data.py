

def visualize_data(file):
    import pandas as pd
    import matplotlib.pyplot as plt
    training_data = pd.read_csv(file)
    pd.set_option('display.max_columns', None)
    print(training_data.columns)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, num="Visualize Data")  # Create the first figure with two plots

    ax1.plot(training_data.index, training_data['Open'], label='Open', color='red')
    ax2.plot(training_data.index, training_data['High'], label='High', color='orange')
    ax3.plot(training_data.index, training_data['Low'], label='Low', color='yellow')
    ax4.plot(training_data.index, training_data['Close'], label='Close', color='blue')
    ax5.plot(training_data.index, training_data['Volume_(BTC)'], label='Volume_(BTC)', color='purple')
    ax6.plot(training_data.index, training_data['Volume_(Currency)'], label='Volume_(Currency)', color='black')
    ax7.plot(training_data.index, training_data['Weighted_Price'], label='Weighted_Price', color='brown')

    ax1.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax2.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax3.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax4.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax5.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax6.legend(bbox_to_anchor=(0, 1.1), loc='upper left')
    ax7.legend(bbox_to_anchor=(0, 1.1), loc='upper left')

    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_data("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")