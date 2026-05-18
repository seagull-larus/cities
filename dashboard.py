import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Устойчивость городской среды", layout="wide")

# Списки признаков по направлению их влияния на городскую среду
STIMULATORS = [
    'people', 'net_salary', 'birth', 'preschool_child', 'length_of_roads',
    'energy_consumption', 'migration_net', 'housing_stock', 'wage', 'workers', 'housing_price'
]

DESTIMULATORS = [
    'air_general_level', 'avg_age', 'ilm', 'crimes', 'criminals', 'death', 'pens', 'poverty_level'
]

@st.cache_data
def load_and_prep_data():
    df_full = pd.read_csv('after_interpolation.csv', encoding='utf-8-sig')
    
    # Собираем только те колонки, которые реально есть в датасете
    feature_columns = [col for col in (STIMULATORS + DESTIMULATORS) if col in df_full.columns]
    
    scalers = {}
    norm_cols = []
    
    # 1. Нормирование с учетом характера влияния признака
    for col in feature_columns:
        if df_full[col].notna().sum() > 0:
            scaler = MinMaxScaler()
            norm_name = f'{col}_norm'
            
            # Базовая нормировка от 0 до 1
            scaled_vals = scaler.fit_transform(df_full[[col]]).flatten()
            
            # Если признак негативный (дестимулятор) — инвертируем его (1 - x)
            if col in DESTIMULATORS:
                df_full[norm_name] = 1.0 - scaled_vals
            else:
                df_full[norm_name] = scaled_vals
                
            scalers[col] = scaler
            norm_cols.append(norm_name)
            
    # 2. Настройка весов признаков (Сумма должна быть равна 1.0)
    # Задаем экспертные веса для ключевых метрик, влияющих на устойчивость
    custom_weights = {
        'poverty_level_norm': 0.12,
        'crimes_norm': 0.12,
        'air_general_level_norm': 0.10,
        'net_salary_norm': 0.08,
        'wage_norm': 0.08,
        'birth_norm': 0.08,
        'death_norm': 0.08,
    }
    
    weights = {}
    # Распределяем остаток весов равномерно между остальными признаками
    missing_cols = [c for c in norm_cols if c not in custom_weights]
    current_sum = sum(custom_weights.get(c, 0) for c in norm_cols if c in custom_weights)
    remainder = max(0.0, 1.0 - current_sum)
    
    for c in norm_cols:
        if c in custom_weights:
            weights[c] = custom_weights[c]
        else:
            weights[c] = remainder / len(missing_cols) if missing_cols else 0
            
    # На всякий случай жестко нормализуем веса в сумму 1.0
    total_w = sum(weights.values())
    weights = {c: w / total_w for c, w in weights.items()}
    
    # 3. Расчет взвешенного интегрального индекса Env Score (шкала от 0 до 100)
    raw_score = sum(df_full[c] * weights[c] for c in norm_cols)

    # 2. Растягиваем его так, чтобы худший результат стал 0, а лучший — 100
    score_min = raw_score.min()
    score_max = raw_score.max()
    
    df_full['env_score'] = ((raw_score - score_min) / (score_max - score_min)) * 100
    
    # Сортировка и создание лагов для ML-модели
    df_full = df_full.sort_values(['city', 'year'])
    for col in norm_cols + ['env_score']:
        df_full[f'lag_{col}'] = df_full.groupby('city')[col].shift(1)
        
    lag_cols = [f'lag_{col}' for col in norm_cols + ['env_score']]
    df_lagged = df_full.dropna(subset=lag_cols)
    
    return df_full, df_lagged, scalers, norm_cols, lag_cols, feature_columns, weights

@st.cache_resource
def train_models(_df_lagged, lag_cols):
    train_data = _df_lagged[_df_lagged['year'] <= 2023]
    test_data = _df_lagged[_df_lagged['year'] == 2024]
    
    X_train, y_train = train_data[lag_cols], train_data['env_score']
    X_test, y_test = test_data[lag_cols], test_data['env_score']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=222, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_r2 = r2_score(y_test, lr.predict(X_test))
    
    return rf, lr, rf_r2, lr_r2

# Загрузка данных и обучение моделей
df_full, df_lagged, scalers, norm_cols, lag_cols, feature_columns, weights = load_and_prep_data()
rf, lr, rf_r2, lr_r2 = train_models(df_lagged, lag_cols)

# Интерфейс Streamlit
st.title("🏙️ AI-Оценка устойчивости городской среды")
st.markdown("Дэшборд для мониторинга и прогнозирования интегрального взвешенного индекса устойчивости городов (Env Score по шкале 0-100).")

latest_year = int(df_full['year'].max())
df_latest = df_full[df_full['year'] == latest_year]
avg_score = df_latest['env_score'].mean()
top_city = df_latest.loc[df_latest['env_score'].idxmax()]

# Главные метрики
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Анализируемых городов", value=len(df_latest))
col2.metric(label=f"Средний Env Score ({latest_year})", value=f"{avg_score:.2f}")
col3.metric(label="Лидер рейтинга", value=top_city['city'], delta=f"{top_city['env_score']:.2f}")
col4.metric(label="Точность прогноза (RF R²)", value=f"{rf_r2:.2f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Обзор и Рейтинги", "📈 Динамика и Признаки", "🔮 AI Сценарный анализ (2025)"])

with tab1:
    st.subheader(f"Топ-15 и Анти-Топ-15 городов по Env Score ({latest_year})")
    
    col_a, col_b = st.columns(2)
    with col_a:
        top_15 = df_latest.nlargest(15, 'env_score').sort_values('env_score', ascending=True)
        fig_top = px.bar(top_15, x='env_score', y='city', orientation='h', title="Лидеры устойчивости", color='env_score', color_continuous_scale='Greens')
        st.plotly_chart(fig_top, use_container_width=True)
        
    with col_b:
        bottom_15 = df_latest.nsmallest(15, 'env_score').sort_values('env_score', ascending=False)
        fig_bottom = px.bar(bottom_15, x='env_score', y='city', orientation='h', title="Аутсайдеры (Зоны риска)", color='env_score', color_continuous_scale='Reds')
        st.plotly_chart(fig_bottom, use_container_width=True)

with tab2:
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("Динамика устойчивости по городам")
        selected_cities = st.multiselect("Выберите города для сравнения:", df_full['city'].unique(), default=['Москва', 'Санкт-Петербург', 'Казань'])
        
        if selected_cities:
            df_trend = df_full[df_full['city'].isin(selected_cities)]
            fig_trend = px.line(df_trend, x='year', y='env_score', color='city', markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
            
    with col_d:
        st.subheader("Важность признаков для предиктивной модели (Random Forest)")
        imp = pd.DataFrame({'Feature': lag_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
        imp['Feature'] = imp['Feature'].str.replace('lag_', '').str.replace('_norm', '')
        fig_imp = px.bar(imp.tail(10), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("Симуляция изменений (Прогноз на следующий период)")
    st.markdown("Изменяйте базовые показатели города, чтобы увидеть, как ML-модель пересчитает рейтинг устойчивости с учетом весов и инверсии негативных факторов.")
    
    col_e, col_f = st.columns([1, 2])
    
    with col_e:
        target_city = st.selectbox("Выберите город для симуляции:", df_latest['city'].unique())
        target_feature = st.selectbox("Выберите показатель для изменения:", feature_columns)
        
        base_data = df_latest[df_latest['city'] == target_city].iloc[0]
        base_val = float(base_data[target_feature])
        
        st.write(f"Текущее значение ({target_feature}): **{base_val:,.2f}**")
        
        percent_change = st.slider("Изменение показателя (%)", min_value=-50, max_value=50, value=0, step=1)
        
        new_val = base_val * (1 + percent_change/100)
        st.write(f"Новое значение: **{new_val:,.2f}**")
        
    with col_f:
        if st.button("Рассчитать AI-прогноз"):
            # Нормируем новое значение через сохраненный MinMaxScaler
            raw_norm = scalers[target_feature].transform([[new_val]])[0, 0]
            
            # Корректно инвертируем, если это негативный фактор
            if target_feature in DESTIMULATORS:
                new_norm = 1.0 - raw_norm
            else:
                new_norm = raw_norm
            
            X_new = {}
            new_norm_values = []
            
            # Формируем измененный лаговый вектор для прогноза
            for c in norm_cols:
                original_feature = c.replace('_norm', '')
                if original_feature == target_feature:
                    X_new[f'lag_{c}'] = new_norm
                    new_norm_values.append(new_norm * weights[c])
                else:
                    X_new[f'lag_{c}'] = base_data[c]
                    new_norm_values.append(base_data[c] * weights[c])
            
            # Рассчитываем взвешенный лаговый env_score по шкале 0-100
            new_env = sum(new_norm_values) * 100
            X_new['lag_env_score'] = new_env
            
            df_pred = pd.DataFrame([X_new])
            pred_rf_new = rf.predict(df_pred)[0]
            
            # Формируем базовый лаговый вектор (без изменений)
            X_base = {f'lag_{c}': base_data[c] for c in norm_cols}
            X_base['lag_env_score'] = base_data['env_score']
            pred_rf_base = rf.predict(pd.DataFrame([X_base]))[0]
            
            diff = pred_rf_new - pred_rf_base
            
            st.success("Прогноз успешно сгенерирован!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Текущий Env Score", f"{base_data['env_score']:.2f}")
            m2.metric("Базовый прогноз (без изм.)", f"{pred_rf_base:.2f}")
            m3.metric("Новый AI-Прогноз", f"{pred_rf_new:.2f}", delta=f"{diff:+.2f}")
            
            # Динамическая подсказка в зависимости от типа признака
            is_destimulator = target_feature in DESTIMULATORS
            behavior = "снижение" if is_destimulator else "рост"
            if percent_change < 0:
                behavior = "рост" if is_destimulator else "снижение"
                
            st.info(f"💡 Изменение показателя `{target_feature}` на {percent_change}% означает {behavior} качества среды. "
                    f"Это приводит к изменению прогнозного индекса устойчивости на {diff:+.2f} пунктов.")

st.caption("Данные подготовлены на основе датасета after_interpolation.csv с применением взвешенной эколого-социальной нормировки.")
