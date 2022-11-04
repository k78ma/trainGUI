import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Vision():

    def __init__(self,dataset):
        self.dataset=dataset

    def bar_plot(self):
        target_name="Employee2010"
        df=self.dataset[target_name]
        df=pd.cut(df,bins=np.arange(0,200,10))
        df.value_counts().plot(kind="bar",title=target_name)
        plt.bar(x=np.arange(0,200,10)[1:]/2,height=df.value_counts(sort=False).values)
        plt.title(target_name)
        plt.xlabel("Employees Number")
        plt.show()


    def run(self):
        self.bar_plot()