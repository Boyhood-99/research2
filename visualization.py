import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./log07-05.csv')
ls = list(df['global_accuracy'])
print(ls)
print(df['global_accuracy'])
fig, ax = plt.subplots()
ax1, = ax.plot(ls)
plt.savefig('fig1.png')
plt.show()
