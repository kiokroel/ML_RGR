import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf


def main_page():
    st.title("Разработка дашборда для моделей ML и анализа данных")

    st.header("Автор:")
    st.write("ФИО: Корень Николай Петрович")
    st.write("Группа: ФИТ-221")
    #st.image("data/images/photo.jpg", width=200)


def about_df_page():
    st.title("Информация о наборе данных:")
    st.header("Мошенничество с кредитными картами")
    st.header("Описание признаков:")
    st.write("- distance_from_home - расстояние от дома до места, где произошла транзакция.")
    st.write("- distance_from_last_transaction - расстояние от места последней транзакции.")
    st.write("- repeat_retailer - произошла ли транзакция у того же розничного продавца.")
    st.write("- ratio_to_median_purchase_price - отношение цены покупки к средней цене покупок.")
    st.write("- used_chip - Это транзакция с помощью чипа(кредитной карты).")
    st.write("- used_pin_number - произошла ли транзакция с использованием PIN-кода")
    st.write("- online_order - является ли транзакция онлайн-заказом.")
    st.write("- fraud - является ли транзакция мошеннической.")


def visualization_page():
    st.title('Визуализация')
    st.title("Датасет мошенничество с кредитными картами")
    df = pd.read_csv('data/df/card_transdata.csv')

    graphics = {
        'Корреляция': __corr_gr,
        'Гистограммы': __histograms,
        'Ящик с усами': __box_plot,
        'Круговая диаграмма': __radial,
    }

    graphics_choice = st.multiselect('Выберите графики', graphics.keys())

    for graphic in graphics_choice:
        graphics[graphic](df)


def predict_page():
    st.title("Предсказание")

    st.header('Загрузите датасет для предсказаний')
    uploaded_file = st.file_uploader("Загрузка файла")

    st.header('Или укажите признаки')

    distanse_from_home = st.number_input("Расстояние от дома до места, где произошла транзакция:", value=0)

    distanse_from_last_transaction = st.number_input("Расстояние от места последней транзакции:", value=0)

    repeat_retailer = st.checkbox("Произошла ли транзакция у того же розничного продавца", value=False)
    repeat_retailer = float(repeat_retailer == True)

    ratio_to_median_purchase_price = st.number_input("Отношение цены покупки к средней цене покупок:", value=0)

    use_chip = st.checkbox("Это транзакция с помощью чипа(кредитной карты)", value=False)
    use_chip = float(use_chip == True)

    used_pin_number = st.checkbox("Произошла ли транзакция с использованием PIN-кода", value=False)
    used_pin_number = float(used_pin_number == True)

    online_order = st.checkbox("Является ли транзакция онлайн-заказом", value=False)
    online_order = float(online_order == True)

    models = {
        'Дерево решений': 'models/Tree_classifier/Tree_classifier.sav',
        'Кластеризация': 'models/KMeans/KMeans.sav',
        'Градиентный бустинг': 'models/GradientBoostingClassifier/GradientBoostingClassifier.sav',
        'Бэггинг': 'models/BaggingClassifier/BaggingClassifier.sav',
        'Стэкинг': 'models/StackingClassifier/StackingClassifier.sav',
        'Нейронная сеть': 'models/NNClassificationModel',
    }

    st.header("Выбор моделей")
    models_choice = st.multiselect('Модели', models.keys())

    button_clicked = st.button("Предсказать")

    if button_clicked:
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            # del data['fraud']
        else:
            data = pd.DataFrame({'distance_from_home': [distanse_from_home],
                    'distance_from_last_transaction': [distanse_from_last_transaction],
                    'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                    'repeat_retailer': [repeat_retailer],
                    'used_chip': [use_chip],
                    'used_pin_number': [used_pin_number],
                    'online_order': [online_order],
                    })

        for model in models_choice:
            if model == 'Нейронная сеть':
                load_model = tf.keras.models.load_model(models[model])
            else:
                load_model = pickle.load(open(models[model], 'rb'))
            st.header(model)
            st.write(load_model.predict(data))


def __corr_gr(df):
    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)


def __histograms(df):
    st.header("Гистограммы")

    columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df.sample(5000)[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)


def __box_plot(df):
    columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
    st.header("Ящики с усами ")
    outlier = df[columns]
    Q1 = outlier.quantile(0.25)
    Q3 = outlier.quantile(0.75)
    IQR = Q3 - Q1
    data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) | (outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data_filtered[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)


def __radial(df):
    st.header("Круговая диаграмма целевого признака")
    plt.figure(figsize=(8, 8))
    df['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('fraud')
    plt.ylabel('')
    st.pyplot(plt)
