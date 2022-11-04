import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Vision():

    def __init__(self,dataset):
        self.dataset=dataset

    def bar_plot_employee_nbr(self):
        target_name="Employee2018"
        df=self.dataset[target_name]
        df.value_counts().plot(kind="bar", title=target_name)
        plt.xscale("symlog")
        plt.xlabel("Employees Number")
        plt.show()
        df=pd.cut(df,bins=np.arange(0,300,5))
        plt.bar(x=np.arange(0,300,5)[1:],height=df.value_counts(sort=False).values)
        plt.title(target_name)
        plt.xlabel("Employees Number")
        plt.show()

    def bar_plot_rev(self):
        target_name="Revenue2017"
        factor=10000
        max=1000
        interval=100
        df=self.dataset[target_name]
        print(df.describe(include = 'all').transpose())
        df=df/factor
        df = pd.cut(df, bins=np.arange(0, max, interval))
        plt.bar(x=list(range(len(np.arange(0, max, interval)[1:]))), height=df.value_counts(sort=False).values)
        plt.xticks(list(range(len(np.arange(0, max, interval)[1:]))),np.arange(0, max, interval)[1:])
        plt.xlabel("Revenue * 10^{}".format(factor))
        plt.title(target_name)
        plt.show()


    def run(self):
        self.bar_plot_rev()