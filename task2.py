import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Дані про продажі за попередні роки
data = {
    'Year': [2019, 2020, 2021, 2022, 2023, 2024],
    'Sales': [150000, 165000, 178000, 185000, 195000, 210000]  # Продажі в гривнях
}

# Створення DataFrame
df = pd.DataFrame(data)

# Перетворення даних
X = df[['Year']]
y = df['Sales']

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Виведення історичних даних для порівняння
print("\nІсторичні дані:")
for year, sales in zip(df['Year'], df['Sales']):
    print(f"{year}: {sales:,} грн")

print(f"Коефіцієнт детермінації (R²): {r2:.2f}")

# Прогноз на 2025 та 2030 роки
prediction2025 = model.predict([[2025]])
prediction2030 = model.predict([[2030]])

print(f"\nПрогноз продажів на 2025 рік: {round(prediction2025[0])} грн")
print(f"Прогноз продажів на 2030 рік: {round(prediction2030[0])} грн")

# Додатковий аналіз
yearly_growth = model.coef_[0]
print(f"\nСередній річний приріст продажів: {round(yearly_growth)} грн")


plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Sales'], color='blue', label='Історичні дані')
plt.plot(df['Year'], model.predict(X), color='red', label='Лінія тренду')

# Додавання прогнозних точок
future_years = [2025, 2030]
future_predictions = [prediction2025[0], prediction2030[0]]
plt.scatter(future_years, future_predictions, color='green', label='Прогноз')

plt.title('Історичні дані та прогноз продажів')
plt.xlabel('Рік')
plt.ylabel('Продажі (грн)')
plt.legend()
plt.grid(True)
plt.show()
