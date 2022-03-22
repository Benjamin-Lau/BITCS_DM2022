# 互评作业1说明

## 数据集

- Wine Reviews
- Oakland Crime Statistics 2011 to 2016

## 使用说明

Python版本：Python 2.7.17

操作系统：Ubuntu

代码的代码的361～368分别是对两个数据集的csv文件的预处理：

```python
wineReviews('winemag-data_first150k.csv')
# wineReviews('winemag-data-130k-v2.csv')
# oaklandCrimeStatistics('records-for-2011.csv')
# oaklandCrimeStatistics('records-for-2012.csv')
# oaklandCrimeStatistics('records-for-2013.csv')
# oaklandCrimeStatistics('records-for-2014.csv')
# oaklandCrimeStatistics('records-for-2015.csv')
# oaklandCrimeStatistics('records-for-2016.csv')
```

选择csv文件对应的语句，取消注释，命令行中输入以下命令来运行：

```bash
mkdir figs
rm ./figs/*.jpg
python preprocessing.py
```

