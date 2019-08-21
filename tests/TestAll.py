import os
directory = os.path.dirname(os.path.abspath(__file__))
os.system('python {}/EuroTest.py'.format(directory))
os.system('python {}/AmericanTest.py'.format(directory))
os.system('python {}/PCATest.py'.format(directory))
os.system('python {}/RegressionTest.py'.format(directory))