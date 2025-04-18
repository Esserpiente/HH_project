import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

hh_data = pd.read_csv ('/Users/alex/Desktop/Python_projects/Head_hunter/hh_database.csv', sep = ';')

# ОБЩАЯ ИНФОРМАЦИЯ О ТАБЛИЦЕ

'''print (hh_data.head())
print (hh_data.tail())
print (hh_data.info()) # Информация о не пустых значениях в столбцах и типах данных
print (hh_data.describe()) # Основная статистическая информация'''


# ПРЕДОБОРАБОТКА ДАННЫХ

#Преобразование признака «Образование и ВУЗ»

education = ['высшее', 'специальное', 'среднее']
hh_data['Образование'] = hh_data['Образование и ВУЗ'].apply (lambda x: x.split(' '))
hh_data['Образование'] = hh_data['Образование'].apply (lambda x: x[:2])
hh_data['Образование'] = hh_data['Образование'].apply (lambda x: x[0] if x[1] not in education else x)
hh_data['Образование'] = hh_data['Образование'].apply (lambda x: ' '.join(x) if type(x)==list else x)
#print (hh_data['Образование'].unique()) #Проверяем полученные категории
#print (hh_data['Образование'].value_counts(ascending=False)) #Считаем кол-во людей
hh_data = hh_data.drop('Образование и ВУЗ', axis= 1) # Удаляем столбец

# Преобразование признака 'Пол, возраст'
hh_data['Пол, возраст'] = hh_data['Пол, возраст'].apply (lambda x: x.split(' , ')) # Преобразуем строку в список
hh_data['Пол'] = hh_data['Пол, возраст'].apply (lambda x: 'М' if x[0] == 'Мужчина' else 'Ж') # Создаем призкак 'Пол'
hh_data['Возраст'] = hh_data['Пол, возраст'].apply (lambda x: x[1].split (' ')) # Создаем признак 'Возраст' и пеобразуем строку в список
hh_data['Возраст'] = hh_data['Возраст'].apply (lambda x: x[1].lstrip()) # Удаляем пробел вначале строки
hh_data['Возраст'] = hh_data['Возраст'].apply (lambda x: int (x)) # Преобразуем строку в число
hh_data = hh_data.drop('Пол, возраст', axis=1) # Удаляем столбец 'Пол, возраст'
#print (round (hh_data['Пол'].value_counts('mean')*100, 2)) # Выводим процентное соотношение резюме женщин и мужчин
#print (hh_data['Возраст'].mean()) # Выводим средний возраст соискателей

# Преобразоыание признака «Опыт работы»

hh_data['Опыт работы'] = hh_data['Опыт работы'].apply (lambda x: np.nan if x=='Не указано' else x) # Вместо 'Не указано' присваиваем NaN
hh_data['Опыт работы'] = hh_data['Опыт работы'].apply (lambda x: x.split (' ') if type (x)== str else np.nan) # Преобразуем строку в список

def job_expirience (exp): # Вычисляем кол-во месяцев 
    month_key_words = ['месяц', 'месяцев', 'месяца'] 
    year_key_words = ['год', 'лет', 'года']
    if type (exp) == list: # Если не NaN тогда считаем стаж работы
        month = 0
        year = 0
        exp = exp[2:6] # Обрезаем не интересущие нас данные
        for i in range (0, len(exp)):
            if exp[i] in month_key_words:
                month = int (exp[i-1])
            if exp[i] in year_key_words:
                year = int (exp[i-1])
        return year*12+month
    else:
        return np.nan
    
hh_data['Опыт работы (месяц)'] = hh_data['Опыт работы'].apply(job_expirience) # Создаем новый признак 'Опыт работы (месяц)'
hh_data = hh_data.drop('Опыт работы',axis=1) # Удаляем столбец 'Опыт работы'
#print (hh_data['Опыт работы (месяц)'].median())


# Преобразоыание признака «Город, переезд, командировки»
interesting_cities = ['Москва', 'Санкт-Петербург'] # Список интересующих нас городов
million_cities = ['Новосибирск', 'Екатеринбург', 'Нижний Новгород', 'Казань',
                  'Челябинск', 'Омск', 'Самара', 'Ростов-на-Дону', 'Уфа',
                  'Красноярск', 'Пермь', 'Воронеж', 'Волгоград' ] # Список городов миллионников

hh_data['Город, переезд, командировки'] = hh_data['Город, переезд, командировки'].apply (lambda x: x.split (' , ')) # Преобразуем строку в список
hh_data['Город, переезд, командировки'] = hh_data['Город, переезд, командировки'].apply (lambda x: x[0:1:]+x[2:] if 'м.' in x[1] else x) # Удаляем данные связанные с метро

hh_data['Город'] = hh_data['Город, переезд, командировки'].apply(lambda x: x[0] if x[0] in interesting_cities       # Создаем новый признак 'Город'
                                                                 else 'город-миллионник' if x[0] in million_cities
                                                                 else 'другие')

hh_data['Готовность к переезду'] = hh_data['Город, переезд, командировки'].apply(lambda x: False if 'не готов' in x[1] else True ) # Создаем новый ризнак 'Готовность к переезду'
hh_data['Готовность к командировкам'] = hh_data['Город, переезд, командировки'].apply(lambda x: False if len (x)>2 and 'не готов' in x[2] else True) # Создаем новый ризнак 'Готовность к командировкам'
hh_data = hh_data.drop('Город, переезд, командировки', axis=1) # Удаляем столбец 'Город, переезд, командировки'

#print (hh_data['Город'].value_counts('mean')) 
mask1 = hh_data['Готовность к переезду']==True
mask2 = hh_data['Готовность к командировкам']==True
#print (round(hh_data.index[(mask1)&(mask2)].value_counts().sum()/hh_data.shape[0]*100)) # Узнаем % готовых к переезду и командировкам от общего кол-ва людей 


# Перобразование признаков 'Занятость' и 'График'


hh_data['Занятость'] = hh_data['Занятость'].apply (lambda x: x.split(', ')) # Преобразуем строку в список
hh_data['частичная занятость'] = hh_data['Занятость'].apply(lambda x: True if 'частичная занятость' in x else False) # Создаем признак 'частичная занятость'
hh_data['проектная работа'] = hh_data['Занятость'].apply(lambda x: True if 'проектная работа' in x else False) # Создаем признак 'проектная работа'
hh_data['полная занятость'] = hh_data['Занятость'].apply(lambda x: True if 'полная занятость' in x else False) # Создаем признак 'полная занятость'
hh_data['стажировка'] = hh_data['Занятость'].apply(lambda x: True if 'стажировка' in x else False) # Создаем признак 'стажировка'
hh_data['волонтерство'] = hh_data['Занятость'].apply(lambda x: True if 'волонтерство' in x else False) # Создаем признак 'волонтерство'

hh_data['График'] = hh_data['График'].apply(lambda x: x.split(', ')) # Преобразуем строку в список
hh_data['полный день'] = hh_data['График'].apply (lambda x: True if 'полный день' in x else False) # Создаем признак 'полный день'
hh_data['сменный график'] = hh_data['График'].apply (lambda x: True if 'сменный график' in x else False) # Создаем признак 'сменный график'
hh_data['гибкий график'] = hh_data['График'].apply (lambda x: True if 'гибкий график' in x else False) # Создаем признак 'гибкий график'
hh_data['удалённая работа'] = hh_data['График'].apply (lambda x: True if 'удалённая работа' in x else False) # Создаем признак 'удалённая работа'
hh_data['вахтовый метод'] = hh_data['График'].apply (lambda x: True if 'вахтовый метод' in x else False) # Создаем признак 'вахтовый метод'

hh_data = hh_data.drop('Занятость', axis=1) # Удаляем столбец 'Занятость'
hh_data = hh_data.drop('График',axis=1) # Удаляем столбец 'График'

mask3 = hh_data['волонтерство']==True
mask4 = hh_data['проектная работа']==True
#print (hh_data.index[(mask3)&(mask4)].value_counts().sum())
    
mask5 = hh_data['вахтовый метод'] == True
mask6 = hh_data['гибкий график'] == True
#print (hh_data.index[(mask5)&(mask6)].value_counts().sum())


# Перобразование признака 'ЗП'
zp_data = pd.read_csv('/Users/alex/Desktop/Python_projects/Head_hunter/ExchangeRates.csv', sep = ',')
hh_data['Обновление резюме'] = pd.to_datetime(hh_data['Обновление резюме'], format='%d.%m.%Y %H:%M') # Переводим в формат datetime 
hh_data['Обновление резюме'] = hh_data['Обновление резюме'].dt.date # Выделяем дату

zp_data['date'] = pd.to_datetime(zp_data['date'], format='%d/%m/%y') # Переводим в формат datetime 
zp_data['date'] = zp_data['date'].dt.date # Выделяем дату

hh_data['ЗП'] = hh_data['ЗП'].apply(lambda x: x.split(' ')) # Преобразуем строку в список
hh_data['Валюта'] = hh_data['ЗП'].apply(lambda x: x[1]) # Создаем признак 'Валюта'
hh_data['Желаемая ЗП'] = hh_data['ЗП'].apply(lambda x: int (x[0])) # Создаем признак 'Желаемая ЗП'

def correct_currency (arg): # Функция преобразования наименования валюты в наименование в ISO-кодировке
    dict_correct_currency = {'грн.':'UAH', 'USD':'USD', 'EUR':'EUR', 'бел.руб.':'BYN', 'KGS':'KGS', 'сум':'UZS', 'AZN':'AZN', 'KZT':'KZT', 'руб.':'руб.'}
    return dict_correct_currency[arg]

hh_data['Валюта'] = hh_data['Валюта'].apply(correct_currency) # Присваиваем наименование валюты в соответствии с ISO-кодировкой

hh_df = pd.merge(hh_data, zp_data, left_on = ['Обновление резюме','Валюта'] , right_on=['date','currency'], how='left') # Объединяем таблицы

hh_df['close'] = hh_df['close'].apply(lambda x: 1 if np.isnan(x)==True else x) # Заменяем NaN на 1 в колонке 'close'
hh_df['proportion'] = hh_df['proportion'].apply (lambda x: 1 if np.isnan(x)==True else x) # Заменяем NaN на 1 в колонке 'proportion'
hh_df['ЗП (руб)'] = hh_df['Желаемая ЗП']*hh_df['close']/hh_df['proportion'] # Создаем новый признак 'ЗП (руб)'

# Удаляем лишние признаки
hh_df = hh_df.drop('ЗП', axis=1)
hh_df = hh_df.drop('Валюта', axis=1)
hh_df = hh_df.drop('Желаемая ЗП',axis=1)
hh_df = hh_df.drop('currency',axis=1)
hh_df = hh_df.drop('per',axis=1)
hh_df = hh_df.drop('date',axis=1)
hh_df = hh_df.drop('time',axis=1)
hh_df = hh_df.drop('close',axis=1)
hh_df = hh_df.drop('vol',axis=1)
hh_df = hh_df.drop('proportion',axis=1)
#print (hh_df['ЗП (руб)'].median())



# ИССЛЕДОВАНИЕ ЗАВИСИМОСТЕЙ В ДАННЫХ

# Распределение признака 'Возраст'

# Строим графики
fig = make_subplots(rows=1, cols=2, subplot_titles=('Коробчатая диаграмма', 'Гистограмма')) # Строим 2 графика рядом 

fig.add_trace(go.Box(x=hh_df['Возраст'], name='Год (лет)'), row=1, col=1) # Строим коробчатую диаграмму
fig.add_trace(go.Histogram(x=hh_df['Возраст'], name='Год (лет)'), row=1, col=2) # Строим гистограмму
# Подписываем графики
fig.update_xaxes(title_text='Возраст', row=1, col=1) 
fig.update_xaxes(title_text='Возраст', row=1, col=2)
fig.update_yaxes(title_text = 'Количество соискателей', row=1, col=2)

fig.update_layout(title_text='Графики распределения признака "Возраст"', #Подписываем subplot
                  title_x= 0.5,
                  legend=dict(x=1, orientation='v')
                 )
#fig.show()
#print('Модальное значение возраста соискателей =', hh_df['Возраст'].mode()[0]) # Модальное значение возраста соискателей


# Распределение признака 'Опыт работы (месяц)'
fig = make_subplots(rows=1, cols=2, subplot_titles=('Коробчатая диаграмма', 'Гистограмма')) # Строим 2 графика рядом 

fig.add_trace(go.Box(x=hh_df['Опыт работы (месяц)'], name='Месяцев'), row=1, col=1) # Строим коробчатую диаграмму
fig.add_trace(go.Histogram(x=hh_df['Опыт работы (месяц)'], name='Месяцев'), row=1, col=2) # Строим гистограмму
# Подписываем графики
fig.update_xaxes(title_text='Опыт работы (месяцев)', row=1, col=1) 
fig.update_xaxes(title_text='Опыт работы (месяцев)', row=1, col=2)
fig.update_yaxes(title_text = 'Количество соискателей', row=1, col=2)

fig.update_layout(title_text='Графики распределения признака "Опыт работы (месяц)"', #Подписываем subplot
                  title_x= 0.5,
                  legend=dict(x=1, orientation='v')
                 )
#fig.show()
#print(hh_df['Опыт работы (месяц)'].mode()) # Модальное значение

# Распределение признака 'ЗП (руб)'
fig = make_subplots(rows=1, cols=2, subplot_titles=('Коробчатая диаграмма', 'Гистограмма')) # Строим 2 графика рядом 

fig.add_trace(go.Box(x=hh_df['ЗП (руб)'], name='RUB'), row=1, col=1) # Строим коробчатую диаграмму
fig.add_trace(go.Histogram(x=hh_df['ЗП (руб)'], name='RUB'), row=1, col=2) # Строим гистограмму
# Подписываем графики
fig.update_xaxes(title_text='Рублей', row=1, col=1) 
fig.update_xaxes(title_text='Рублей', row=1, col=2)
fig.update_yaxes(title_text = 'Количество соискателей', row=1, col=2)

fig.update_layout(title_text='Графики распределения признака "ЗП (руб)"', #Подписываем subplot
                  title_x= 0.5,
                  legend=dict(x=1, orientation='v')
                 )
#fig.show()

# Диаграмма зависимости медианной желаемой заработной платы («ЗП (руб)») от уровня образования («Образование»)

users_salary = hh_df[hh_df['ЗП (руб)']<1000000] # Отсеиваем тех у кого ЗП больше 1000000
users_education = users_salary.groupby(by=['Образование'],as_index=False)['ЗП (руб)'].median() # Группируем и вычисляем медианное значение
# Строим график
fig = go.Figure()
fig.add_trace(go.Bar(x=users_education['Образование'], y = users_education['ЗП (руб)']))
fig.update_layout(
    title='Зависимость медианной желаемой заработной платы от уровня образования',
    title_x = 0.5,
    xaxis_title='Уровень образования',
    yaxis_title='Заработная плата (руб)',
    )
#fig.show()

# Диаграмма распределения желаемой заработной платы («ЗП (руб)») в зависимости от города («Город»)
# Строим график
fig = go.Figure()
fig.add_trace(go.Box(y = users_salary['Город'], x=users_salary['ЗП (руб)'])) # Строим график зависимости по ранее отфильтрованным данным users_salary
fig.update_traces(orientation='h')
fig.update_layout(
    title='Распределения желаемой заработной платы в зависимости от города',
    title_x = 0.5,
    xaxis_title='Заработная плата (руб)',
    yaxis_title='Наименование городов'
    )   
#fig.show()

# Многоуровневая столбчатая диаграмма, которая показывает зависимость медианной заработной платы («ЗП (руб)»)
#от признаков «Готовность к переезду» и «Готовность к командировкам»

ready_to_move = hh_df.groupby(by=['Готовность к переезду'],as_index=False)['ЗП (руб)'].median() # Группируем и вычисляем медианное значение
ready_to_trip = hh_df.groupby(by=['Готовность к командировкам'],as_index=False)['ЗП (руб)'].median() # Группируем и вычисляем медианное значение

# Строим график
fig = go.Figure()
fig.add_trace(go.Bar(x=ready_to_move['Готовность к переезду'], y = ready_to_move['ЗП (руб)'], name = 'Переезд'))
fig.add_trace(go.Bar(x=ready_to_trip['Готовность к командировкам'], y = ready_to_trip['ЗП (руб)'], name = 'Командировки'))
fig.update_layout(
    title='Зависимость медианной заработной платы от от признаков «Готовность к переезду» и «Готовность к командировкам»',
    title_x = 0.5,
    xaxis_title='Готовность',
    yaxis_title='Заработная плата (руб)',
    )
#fig.show()

# Тепловая карта, иллюстрирующая зависимость медианной желаемой заработной платы от возраста и образования

pivot = hh_df.pivot_table( # Группируем и вычисляем медианное значение (pivot для того чтобы пропуски 0 заполнить)
    values='ЗП (руб)',
    index='Образование',
    columns='Возраст',
    aggfunc='median',
    fill_value= 0 # Заполняем пропуски
)
#Строим график
fig = go.Figure()
fig.add_trace(go.Heatmap(x =pivot.columns, y =pivot.index, z=pivot.values))
fig.update_layout(
    title='Тепловая карта, иллюстрирующая зависимость медианной желаемой заработной платы от возраста и образования',
    title_x = 0.5,
    xaxis_title='Возраст',
    yaxis_title='Тип образования',
    )
#fig.show()

# Диаграмма рассеяния, показывающая зависимость опыта работы (в годах) от возраста

hh_df['Опыт работы (лет)'] = hh_df['Опыт работы (месяц)'].apply (lambda x: x/12) #Создаем признак Опыт работы, вырвженный в годах 
#Строим график
fig = go.Figure()    

fig.add_trace(go.Scatter(x = hh_df['Возраст'], y = hh_df['Опыт работы (лет)'], mode='markers', name = 'Опыт'))
fig.add_trace(go.Scatter(x = [0, 100], y = [0, 100], name = 'Опыт работы = Возраст'))
fig.update_layout(
    title='Зависимость опыта работы (в годах) от возраста',
    title_x = 0.5,
    xaxis_title='Возраст соискателей',
    yaxis_title='Опыт работы (лет)',
)
#fig.show()
hh_df = hh_df.drop('Опыт работы (лет)', axis= 1) #Удаляем ранее созданный признак

# ДОПОЛНИТЕЛЬНЫЕ БАЛЛЫ

#Зависимость средней ЗП от наличия автомобиля
auto = hh_df.groupby(by = ['Авто'], as_index=False)['ЗП (руб)'].mean() # Группируем интересующие данные и вычисляем среднее значение
auto['Авто'] = auto['Авто'].apply (lambda x: 'Нет автомобиля' if x=='Не указано' else x) # Меняем "Не указано" на "Нет автомобиля"

#Строим график
fig = go.Figure()
fig.add_trace(go.Bar( x= auto['Авто'], y = auto ['ЗП (руб)']))
fig.update_layout(
    title='Зависимость средней ЗП от наличия автомобиля',
    title_x = 0.5,
    xaxis_title='Наличие автомобиля',
    yaxis_title='Заработная плата (руб)',
    )
#fig.show()


# Зависимость средней ЗП от признака "Пол"
male_female = hh_df.groupby(by=['Пол'], as_index=False)['ЗП (руб)'].mean() # Группируем интересующие данные и вычисляем среднее значение
# Строим график
fig = go.Figure()
fig.add_trace(go.Bar( x= male_female['Пол'], y = male_female ['ЗП (руб)']))
fig.update_layout(
    title='Зависимость средней ЗП от пола соискателя',
    title_x = 0.5,
    xaxis_title='Пол соиcкателя',
    yaxis_title='Заработная плата (руб)',
    )
#fig.show()

# ОЧИСТКА ДАННЫХ

#Ищем дубликаты
double_hh_df = list (hh_df.columns)
mask9 = hh_df.duplicated(subset=double_hh_df)
hh_df_dublicates = hh_df[mask9]
#print (f'Количество дубликатов в данных, {hh_df_dublicates.shape[0]}')

# Удаляем дублирующие записи
hh_df = hh_df.drop_duplicates(subset=double_hh_df, ignore_index=True)
#print(f'Результирующее число записей: {hh_df.shape[0]}')

# Ищем пропуски
cols_null = hh_df.isnull().sum() # Ищем столбцы с пропусками
#print (cols_null[cols_null>0].sort_values(ascending=False)) # Выводим только те столбцы где есть пропуски, считаем сколько

# Заполняем пропуски
hh_df = hh_df.fillna({'Опыт работы (месяц)':hh_df['Опыт работы (месяц)'].median()}) # Заполняем пропуски в признаке 'Опыт работы (месяц)' медианным значением
#Удаляем пропуски
hh_df = hh_df.dropna(subset=['Последнее/нынешнее место работы', 'Последняя/нынешняя должность']) # Удаляем пропуски в столбцах 'Последнее/нынешнее место работы', 'Последняя/нынешняя должность'
#print (hh_df['Опыт работы (месяц)'].mean())

# Создаем условие
mask10 = hh_df['ЗП (руб)']<=1000000 
mask11 = hh_df['ЗП (руб)']>=1000
hh_df = hh_df[mask10&mask11] # Отсеиваем лишнее по условию
#print (hh_df.info())

# Отсеиваем тех у кого опыт больше возраста
hh_df['Опыт работы (лет)'] = hh_df['Опыт работы (месяц)'].apply (lambda x: x/12) #Создаем признак Опыт работы, вырвженный в годах 

mask12 = hh_df['Опыт работы (лет)']<hh_df['Возраст']
hh_df = hh_df[mask12]
hh_df = hh_df.drop('Опыт работы (лет)', axis=1) # Удаляем использованный признак
#print(hh_df.info())


# Hаспределение признака 'Возраст' в логарифмическом масштабе

log_age = np.log(hh_df['Возраст']) #Логарифмируем признак
#Строим график
fig = go.Figure()
fig.add_trace(go.Histogram(x=log_age, name='Год (лет)')) # Строим гистограмму
#Строим линии
fig.add_vline(x=log_age.mean(), line_width=3, line_dash='solid', line_color='black') 
fig.add_vline(x=log_age.mean() + 3 * log_age.std(), line_width=3, line_dash='dash', line_color='black')
fig.add_vline(x=log_age.mean() - 3 * log_age.std(), line_width=3, line_dash='dash', line_color='black')
fig.update_layout(title='Графики логарифмического распределения признака "Возраст"',
                  title_x= 0.5,
                  xaxis_title='Возраст соискателя',
                  yaxis_title='Количество соискателей'
                 )
#fig.show()

# Метод z-отклонений (метод 3 сигм)

def outliers_z_score_mod(data, feature, log_scale=False, left = 3, right = 3):
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    
    return outliers, cleaned

outliers, cleaned = outliers_z_score_mod(hh_df, 'Возраст', log_scale=True, right = 4)
#print(f'Число выбросов по методу z-отклонений: {outliers.shape[0]}')
#print(f'Результирующее число записей: {cleaned.shape[0]}')
#print (f'Под выбросы попадают соискатели с возрастом,  {outliers['Возраст']}')
