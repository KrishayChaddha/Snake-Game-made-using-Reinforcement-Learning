import matplotlib.pyplot as plt

def plot_training_progress(scores, average_scores):
    plt.clf()
    plt.title('Training Progress - Deep Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Episode', alpha=0.5, color='cornflowerblue')
    plt.plot(average_scores, label='Cumulative Average', color='firebrick', linewidth=2)
    plt.ylim(bottom=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(average_scores) - 1, average_scores[-1], f"{average_scores[-1]:.1f}")
    plt.legend(loc='upper left')
    plt.draw()
    plt.pause(0.1)