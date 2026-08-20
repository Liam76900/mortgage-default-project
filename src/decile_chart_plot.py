import matplotlib.pyplot as plt

def plot_decile_chart(decile_baseline, decile_balanced):
    plt.figure(figsize=(9,5))
    plt.plot(decile_baseline.index, decile_baseline.values, marker='o', label='Baseline')
    plt.plot(decile_balanced.index, decile_balanced.values, marker='o', label='Balanced')
    plt.xlabel('Predicted Risk Decile (0 = safest, 9 = riskiest)')
    plt.ylabel('Actual Default Rate')
    plt.title('Actual Default Rate by Predicted Risk Decile')
    plt.legend()
    plt.savefig('outputs/decile_chart.png', dpi=150, bbox_inches='tight')
    plt.show()