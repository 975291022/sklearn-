import pandas as pd
def data():
    x = pd.read_csv("feature.csv", header=None)
    y = pd.read_csv("label.csv", header=None)
    data = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
    data.columns = ["Al", 'Ti', 'Li', 'Ge', 'Se', 'label']
    print(data)
    y = y.values.ravel()
    return (x,y)
