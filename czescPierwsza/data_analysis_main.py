from Functions.RequirementsFor3 import LoadingData, CalculateStatisticalFeature
from Functions.RequirementsFor35 import boxplots, ViolinPlots
from Functions.RequirementsFor4 import ErrorBars, Distributions, DistributionsWithParameter
from Functions.RequirementsFor45 import Heatmap


def saveAndCalculate():
    data = LoadingData.LoadData()
    CalculateStatisticalFeature.CalculateStatisticalFeature(data)
    boxplots.plot1(data)
    boxplots.plot2(data)
    boxplots.plot3(data)

    ViolinPlots.plot1(data)
    ViolinPlots.plot2(data)

    ErrorBars.plot1(data)
    ErrorBars.plot2(data)

    Distributions.plot1(data)
    Distributions.plot2(data)
    Distributions.plot3(data)
    Distributions.plot4(data)

    DistributionsWithParameter.plot1(data)
    DistributionsWithParameter.plot2(data)
    DistributionsWithParameter.plot3(data)
    DistributionsWithParameter.plot4(data)

    Heatmap.plot1(data)
    Heatmap.plot2(data)
    Heatmap.plot3(data)
    Heatmap.plot4(data)

if __name__ == '__main__':
    saveAndCalculate()