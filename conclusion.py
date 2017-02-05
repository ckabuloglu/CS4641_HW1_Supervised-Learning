import matplotlib.pyplot as plt
import seaborn as sns
from numpy import arange

sns.set(color_codes=True)

letter = [86.9, 75.4, 26.6, 97.1, 93.9]
nursery = [99.4, 96.9, 80.3, 95.1, 93.6]

kernels = ['Decision Tree','Neural Net', 'Boosting', 'SVM', 'KNN']
temp_x = arange(5)
fig = plt.figure()
plt.style.use('ggplot')
plt.bar(temp_x - 0.22, nursery, width=0.20, color='r', label="Nursery")
plt.bar(temp_x + 0.02, letter, width=0.20, color='b', label="Letter")
plt.xticks(temp_x, kernels)
plt.xlabel('Algorithms')
plt.ylabel('Best Accuracy')
plt.legend(loc='best')
plt.title('Supervised Learning Algorithms Compared')
fig.savefig('figures/conclusion.png')
plt.close(fig)