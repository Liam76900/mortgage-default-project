import matplotlib.pyplot as plt

def plot_ltv_band_chart(ltv_check):
    plt.figure(figsize=(9,5))
    labels = [str(interval) for interval in ltv_check.index]
    plt.bar(labels, ltv_check.values, color='steelblue')
    plt.xlabel('LTV Band')
    plt.ylabel('Average Predicted Default Probability')
    plt.title('Predicted Default Risk by Loan-to-Value Band')
    plt.savefig('outputs/ltv_band_chart.png', dpi=150, bbox_inches='tight')
    plt.show()