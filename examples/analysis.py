# A beginner's analysis with some common mistakes. Try:
#   cd examples && python make_survey.py
#   learnstat check analysis.py      (suggestions, nothing is run)
#   learnstat run analysis.py        (suggestions + guarded error recovery)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv("survey.csv")

print(df["income"].mean())

df["country"].value_counts().plot.pie()
plt.close()

treated = df[df["group"] == "treatment"]["Score"]
control = df[df["group"] == "control"]["score"]
print(stats.ttest_ind(treated, control))

df.groupBy("gender")["income"].median()
