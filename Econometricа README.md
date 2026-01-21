[Econometrics 5 example.py](https://github.com/user-attachments/files/24780322/Econometrics.5.example.py)[Uploading Econometrics 5 example.# Exercise 1

# probit-регрессия
# 
# тестирование значимости k при FAMINC
# 
# Найти: тестовую статистику, критическое значение (при 0.01)

# 1. Вся модель 
#     1) lms (F робастный) 
#     2) log/prob (LR сравнение с моделью с константой)
# 
# -
# 
# 2. 1 переменная
#     1) lms (t робастныйс с ошибками НС3!) 
#     2) log/prob (z)
# 
# -
# 
# 3. 2 переменных
#     1) lms (F робастный Wald) 
#     2) log/prob (LR или Wald)

# Тут 2В

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


mod_prob = smf.probit(formula='LFP~WA+I(WA**2)+CIT+UN+np.log(FAMINC)', data=df) # спецификация модели
res_prob = mod_prob.fit() # подгонка модели

res_prob.summary() # отчет

# коэффициенты подогнанной модели
res_prob.params.round(3)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_params

# уровень значимости 1%

# вывод результатов z-теста
summary_params(res_prob)

#критическое значение

from scipy.stats import norm 

sign_level = 0.01
norm.ppf(q=1-sign_level/2)

#Тестовая статистика

#отвергаем если |z|>zcr коэфф значим
# тестовые z-статистики для кажлого коэффциента с округленим
res_prob.tvalues.round(3)

#узнаем коэффициент
# P-значения для z-статистик с округленим
res_prob.pvalues.round(4)

# k при FAMINC = 0.0001
# 
# тестовая стат = 3.83
# 
# крит = 2.58
# 
# значим



2 b (z) уже рассмотрен


1. Вся модель
a) lms (F робастный)

1. Вся модель
b) log/prob (LR сравнение с моделью с константой)

2. 1 переменная
а) lms (t робастныйс с ошибками НС3!)

3. 2 переменных
 a) lms (F робастный Wald)

3. 2 переменных
в) log/prob (LR или Wald)


1 b - logit регрессия вся модель
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_params
from scipy.stats import t # t-распределение

#Для датасета 
modlr = smf.logit(formula=' LFP~K618+CIT+UN ', data=df) # спецификация модели
reslr = modlr.fit() # подгонка модели
reslr.summary() # отчет

# Число наблюдений, по которым была оценена модель
res.nobs

# логарифм функции правдоподобия для модели
reslr.llf.round(3)

# логарфим функции правдоподобия для регрессии без объясняющих переменных (только на константу)
reslr.llnull.round(3)

# Тестовая статистика LR-теста и её P-значение с округленим, значимость 10%
reslr.llr.round(3), reslr.llr_pvalue.round(3)

# степени свободы хи кадрат
reslr.df_model

from scipy.stats import chi2 

#критическое значение хи хвадрат альфа
sign_level = 0.1 # уровень значимости
chi2.ppf(q=1-sign_level, df=reslr.df_model).round(3) 


Значима ли регрессия? если LRstat > хи квадрат альфа = > отвергаем нулевую гипотезу (регрессия значима) если наоборот = незначима





# Exercise 2
# 
# 
# probit-регресия
# 
# прогноз
# 
# уже есть таблица

import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.iolib.summary2 import summary_col, summary_params

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


# Создаем спецификацию модели через формулу и подгоняем модель с поправкой на гетероскедастичность
tablef5 = smf.probit(formula='LFP~WA+WE+KL6+K618+CIT+UN', data=df).fit(cov_type='HC3')
#print(tablef5.summary(slim=True))
tablef5.params.round(3)

# рассмотрим людей с характеристиками
new_df = pd.DataFrame( {'WA': [34, 39], 
                        'WE': [15, 14], 
                        'KL6': [2, 1], 
                        'K618': [0, 1], 
                      'CIT': [0, 1], 
                     'UN': [5, 2], } )
new_df 

#Прогноз для каждого
tablef5.predict(exog=new_df, transform=True).round(3)








# Exercise 3
# 
# 
# probit-регрессия
# 
# 
# средний (по выборке) предельный эффект для переменной KL6

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


# probit
res_probit = smf.probit(formula='LFP~WA+WE+KL6+K618+CIT+UN', data=df).fit()

margeff_probit = res_probit.get_margeff(at='overall')
# вывод результатов
margeff_probit.summary()
# краткий отчёт
# margeff_probit.summary_frame() 

# Средний по выборке предельный эффект KL6 = -0.3109






Предельный эффект для переменной KL6 в средней точке.

logit-регрессия

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))
#подключим датасет mroz_Greene из локального файла
#df = pd.read_csv('TableF5-1.csv', na_values=(' ', '', '  '))

# LPM 
res_lpm_hc = smf.ols(formula=' LFP~WA+WE+KL6+K618+CIT+UN ', data=df).fit(cov_type='HC3')


# logit
res_logit = smf.logit(formula='LFP~WA+WE+KL6+K618+CIT+UN ', data=df).fit()


# probit
res_probit = smf.probit(formula='LFP~WA+WE+KL6+K618+CIT+UN', data=df).fit()

ПРОГНОЗИРОВАНИЕ
# Создадим датафрейм с новыми данными регрессоров для прогноза
new_data = pd.DataFrame({'WA':[35, 40, 42], 
                         'WE': [15, 12, 14], 
                         'KL6': [2, 1, 2],
                         'K618': [0, 2, 1], 
                         'CIT': [1, 0, 1], 
                         'UN':[5, 7.5, 3]})
new_data

# Прогноз для LPM с округлением до 3-х десятичных знаков
res_lpm_hc.predict(exog=new_data, transform=True).round(3)

# Прогноз для logit с округлением до 3-х десятичных знаков
res_logit.predict(exog=new_data, transform=True).round(3)

# Прогноз для probit с округлением до 3-х десятичных знаков
res_probit.predict(exog=new_data, transform=True).round(3)


Предельные значения для каждого регрессора в средней точке для logit модели

margeff_logit = res_logit.get_margeff(at='mean')
# вывод результатов
margeff_logit.summary()
# краткий отчёт
# margeff_logit.summary_frame() 


Средние по выборке предельные значения для каждого регрессора в средней точке для logit модели
 
margeff_logit = res_logit.get_margeff(at='overall')
# вывод результатов
margeff_logit.summary()
# краткий отчёт
# margeff_logit.summary_frame() 


Предельные значения для каждого регрессора в средней точке для probit модели

margeff_probit = res_probit.get_margeff(at='mean')
# вывод результатов
margeff_probit.summary()
# краткий отчёт
# margeff_probit.summary_frame() 


Средние по выборке предельные значения для каждого регрессора в средней точке для probit модели

margeff_probit = res_probit.get_margeff(at='overall')
# вывод результатов
margeff_probit.summary()
# краткий отчёт
# margeff_probit.summary_frame() 


probit-регрессия: Качество подгонки и Сравнение моделей

# model 
mod = smf.probit(formula = ' LFP~WA+WE+KL6+K618+CIT+UN ', data = df)
res = mod.fit(disp=False)

# порядок регрессоров в таблице
reg_order = ['Intercept', 'WA', 'WE', 'KL6', 'K618', 'CIT','UN']
# Зависимая переменная LFP
summary_col([res], stars=True, regressor_order=reg_order, float_format='%.3f')


дальше всякие псевдо





# Exercise 4

# шесть logit-регрессий
# 
# Для любых регрессий (logit, probit, ols) 
# 
# pseudoR2   - наибольшее     
# 
# 
# pseudoR2-adj    - наибольшее 
# 
# 
# Log Likelihood     - наибольшее   
# 
# 
# Akaike Inf. Crit.  - наименьшее 
# 
# 
# Bayesian Inf. Crit. - наименьшее    

  





# Exercise 5

# LPM-регрессия
# Уровень значимости 5%
# 
# 
# тестовая статистика и критическое значение - ?

# F робастный (см 1 упр)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_params
from scipy.stats import t # t-распределение

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


from statsmodels.iolib.summary2 import summary_col, summary_params # вывод результатов тестирования
from scipy.stats import f 

#зададим спецификацию модели через формулу
mod_lpm_f = smf.ols(formula=' LFP~WA+I(WA**2)+CIT+UN+np.log(FAMINC) ', data=df)

# подгонка модели с поправкой на гетероскедастичность
res_lpm_hc_f = mod_lpm_f.fit(cov_type='HC3')
print(res_lpm_hc_f.summary(slim=True))



#Результаты робастного F-теста (тестовая статистика)

# тестовая статистика и P-значение
np.round(res_lpm_hc_f.fvalue, 3), np.round(res_lpm_hc_f.f_pvalue, 3)


# зададим уровень значимости
alpha = 0.05

#5%-критическое значение F-распределения
f.ppf(q=1-0.05, dfn=res_lpm_hc_f.df_model, dfd=res_lpm_hc_f.df_resid).round(3)

#Критическое 

# можно использовать переменную alpha
f.ppf(q=1-alpha, dfn=res_lpm_hc_f.df_model, dfd=res_lpm_hc_f.df_resid).round(3)

# Вывод: так как F тестовое (4.994) > F крит (2.226) => регрессия значима






 

# Exercise 6

# logit-регрессия и R2pseudo
# 
# (также и для probit)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col # вывод результатов нескольких регрессий

# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


df.head()

mod = smf.logit(formula = ' LFP~WE+KL6+K618+CIT+UN+np.log(FAMINC)', data = df)
res = mod.fit(disp=False)

reg_order = ['Intercept', 'WE', 'KL6', 'K618', 'CIT','UN', 'np.log(FAMINC)']
# Зависимая переменная LFP
summary_col([res], stars=True, regressor_order=reg_order, float_format='%.3f')

# Качество подгонки. Базовые показатели
# McFadden's R2pseudo
# 

# pseudoR2 
res.prsquared.round(3)

# McFadden’s Adjusted R2pseudo.adj

# pseudoR2.adj 
(1-(res.llf-res.df_model-1)/res.llnull).round(3)

# Cox & Snell R2

# Cox.Snell.R2 
(1-np.exp(-res.llr/res.nobs)).round(3)

# Nagelkerke / Cragg & Uhler R2

# Nagelkerke.R2 
((1-np.exp(-res.llr/res.nobs))/(1-np.exp(2*res.llnull/res.nobs))).round(3)


# Efron's R2

# Efron.R2 for model 1
(1-(np.sum(res.resid_response**2))/(res.nobs*np.var(mod.endog))).round(3)

# McKelvey & Zavoina's R2

#McKelvey.Zavoina.R2
y_prob = res.predict(mod.exog, transform=False)

# probit
(np.var(y_prob)/(np.var(y_prob)+1)).round(3)







# Exercise 7

# см упр 1

# probit регрессия
# 
# коэффициент при регрессоре FAMINC? Дать интерпретацию

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


mod_prob = smf.probit(formula=' LFP~WA+I(WA**2)+CIT+UN+np.log(FAMINC)', data=df) # спецификация модели
res_prob = mod_prob.fit() # подгонка модели

res_prob.summary() # отчет

# коэффициенты подогнанной модели
res_prob.params.round

#Не нужно, если просто просят коэффициент

#узнаем коэффициент
# P-значения для z-статистик с округленим
res_prob.pvalues.round(4)

# Интерпретация коэфф при np.log(FAMINC) - 0.362126
# 

# Data description
# 
# loanapp
# 
# occ: occupancy
# loanamt: loan amt in thousands
# action: type of action taken
# msa: msa number of property
# suffolk: =1 if property in suffolk co.
# appinc: applicant income, $1000s
# typur: type of purchaser of loan
# unit: number of units in property
# married: =1 if applicant married
# dep: number of dependents
# emp: years employed in line of work
# yjob: years at this job
# self: =1 if self employed
# atotinc: total monthly income
# cototinc: coapp total monthly income
# hexp: propose housing expense
# price: purchase price
# other: other financing, $1000s
# liq: liquid assets
# rep: no. of credit reports
# gdlin: credit history meets guidelines
# lines: no. of credit lines on reports
# mortg: credit history on mortgage paym
# cons: credit history on consumer stuf
# pubrec: =1 if filed bankruptcy
# hrat: housing exp, percent total inc
# obrat: other oblgs, percent total inc
# fixadj: fixed or adjustable rate?
# term: term of loan in months
# apr: appraised value
# prop: type of property
# inss: PMI sought
# inson: PMI approved
# gift: gift as down payment
# cosign: is there a cosigner
# unver: unverifiable info
# review: number of times reviewed
# netw: net worth
# unem: unemployment rate by industry
# min30: =1 if minority pop. > 30percent
# bd: =1 if boarded-up val > MSA med
# mi: =1 if tract inc > MSA median
# old: =1 if applic age > MSA median
# vr: =1 if tract vac rte > MSA med
# sch: =1 if > 12 years schooling
# black: =1 if applicant black
# hispan: =1 if applicant Hispanic
# male: =1 if applicant male
# reject: =1 if action == 3
# approve: =1 if action == 1 or 2
# mortno: no mortgage history
# mortperf: no late mort. payments
# mortlat1: one or two late payments
# mortlat2: > 2 late payments
# chist: =0 if accnts deliq. >= 60 days
# multi: =1 if two or more units
# loanprc: amt/price
# thick: =1 if rep > 2
# white: =1 if applicant white
# 
# 
# TableF5-1
# 
# 
# Labor Supply Data From Mroz (1987), 753 Observations Source: 1976 Panel Study of Income Dynamics, Mroz(1987).
# 
# LFP = A dummy variable = 1 if woman worked in 1975, else 0
# WHRS = Wife's hours of work in 1975
# KL6 = Number of children less than 6 years old in household
# K618 = Number of children between ages 6 and 18 in household
# WA = Wife's age
# WE = Wife's educational attainment, in years
# WW = Wife's average hourly earnings, in 1975 dollars
# RPWG = Wife's wage reported at the time of the 1976 interview (not = 1975 estimated wage)
# HHRS = Husband's hours worked in 1975
# HA = Husband's age
# HE = Husband's educational attainment, in years
# HW = Husband's wage, in 1975 dollars
# FAMINC = Family income, in 1975 dollars
# WMED = Wife's mother's educational attainment, in years
# WFED = Wife's father's educational attainment, in years
# UN = Unemployment rate in county of residence, in percentage points.
# CIT = Dummy variable = 1 if live in large city (SMSA), else 0
# AX = Actual years of wife's previous labor market experience
# 
# 
# 
# 
# TableF7-3
# Expenditure and Default Data,  Source: Greene (1992)
# 
# Cardhldr = Dummy variable, 1 if application for credit card accepted, 0 if not
# Default = 1 if defaulted 0 if not (observed when Cardhldr = 1, 10,499 observations),
# Age = Age in years plus twelfths of a year,
# Adepcnt = 1 + number of dependents,
# Acadmos = months living at current address,
# Majordrg = Number of major derogatory reports,
# Minordrg = Number of minor derogatory reports,
# Ownrent = 1 if owns their home, 0 if rent
# Income = Monthly income (divided by 10,000),
# Selfempl = 1 if self employed, 0 if not,
# Inc_per = Income divided by number of dependents,
# Exp_Inc = Ratio of monthly credit card expenditure to yearly income,
# Spending = Average monthly credit card expenditure (for Cardhldr = 1),
# Logspend = Log of spending.
# 
# 
#     
# SwissLabour
#     
# Cross-section data originating from the health survey SOMIPOPS for Switzerland in 1981.
# 
# participation = Factor. Did the individual participate in the labor force?
# income = Logarithm of nonlabor income.
# age = Age in decades (years divided by 10).
# education = Years of formal education.
# youngkids = Number of young children (under 7 years of age).
# oldkids = Number of older children (over 7 years of age).
# foreign = Factor. Is the individual a foreigner (i.e., not Swiss)?
# 
# Source: Gerfin, M. (1996). Parametric and Semi-Parametric Estimation of the Binary Response Model of Labour

# Интерпретация ols/logit/probit







# Exercise 8

# LPM-регрессия
# 
# 
# Есть таблица


import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.iolib.summary2 import summary_col, summary_params

# подключим датасет mroz_Greene по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))
#подключим датасет mroz_Greene из локального файла
#df = pd.read_csv('TableF5-1.csv', na_values=(' ', '', '  '))

# Создаем спецификацию модели через формулу и подгоняем модель с поправкой на гетероскедастичность
tablef5 = smf.ols(formula='LFP~WA+I(WA**2)+WE+CIT+UN+np.log(FAMINC)', data=df).fit(cov_type='HC3')
#print(tablef5.summary(slim=True))
tablef5.params.round(3)

# рассмотрим 3 людей с характеристиками
new_df = pd.DataFrame( {'WA': [32, 37, 40], 
                        'WE': [15, 11, 13], 
                        'CIT': [1, 0, 1], 
                        'UN': [3, 5, 7.5], 
                       'FAMINC': [35000, 48500, 67800]} )
new_df 

tablef5.predict(exog=new_df, transform=True).round(3)








# Exercise 9

# probit-регрессия
# Cox & Snell R2

# см 6 упр

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col # вывод результатов нескольких регрессий

# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# подключим датасет по ссылке 
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))


df.head()

mod = smf.probit(formula = '  LFP~WA+I(WA**2)+FAMINC+I(FAMINC**2)+KL6+K618+CIT+UN ', data = df)
res = mod.fit(disp=False)

reg_order = ['Intercept', 'WA', 'I(WA**2)','KL6', 'K618', 'CIT','UN', 'np.log(FAMINC)']
# Зависимая переменная LFP
summary_col([res], stars=True, regressor_order=reg_order, float_format='%.3f')

# Cox.Snell.R2 
(1-np.exp(-res.llr/res.nobs)).round(3)






# Exercice 10

# logit-регрессия
# 
# совместная значимость CIT, UN, np.log(FAMINC)
# 
# использовать тест Вальда
# 
# Уровень значимости 1%

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Не показывать FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# импорт датасета
df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/econometrica/refs/heads/main/econometrica-2/datasets/TableF5-1.csv', na_values=(' ', '', '  '))
# импорт данных из локального файла
# df = pd.read_csv('loanapp.csv')
df

mod_full = smf.logit(formula='LFP~WA+I(WA**2)+WE+KL6+K618+CIT+UN+np.log(FAMINC)', data=df) # спецификация модели
res_full = mod_full.fit() # подгонка модели

#wald_test для всех

wald_test=res_full.wald_test ('(CIT=0, UN=0, np.log(FAMINC)=0)')

# тестовая статистика хи2
test_stat=wald_test.statistic[0][0]
test_stat_rounded=round(test_stat, 2) 
print (f"Тестовая статистика = {test_stat_rounded}")


# степень свободы 3 (3 переменные)
df_wald=3

# критическое значение хи2 (3) для альфа=0.01
alpha=0.01
chi2_crit=chi2.ppf(1-alpha, df_wald)
chi2_crit_rounded=round(chi2_crit,2)
print (f"Критическое значение = {chi2_crit_rounded}")























py…]()
