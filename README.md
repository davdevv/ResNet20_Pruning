# Filter Pruning using clustering method

* models - папка с моделью ResNet20.py и весами best_test_acc.pth
* data - папка с cifar10.py, который упрощает работу с датасетом и аугментациями augmentation.py
* train.py - содержит функцию train для обучения модели и validate для вычислений метрик на validation
* pruning.py - скрипт, реализующий прунинг
* utils.py - функции для отображения графиков и сохранения весов
* overview.ipynb - обзор полученных результатов
