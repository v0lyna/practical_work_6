import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDataAnalyzer:
    """Клас для аналізу медичних даних пацієнтів."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        np.random.seed(seed)

    @staticmethod
    def generate_sample_data(n_samples: int = 299) -> pd.DataFrame:
        """Генерація тестового набору даних."""
        return pd.DataFrame({
            "вік": np.random.randint(30, 80, size=n_samples),
            "анемія": np.random.randint(0, 2, size=n_samples),
            "креатинін_фосфокіназа": np.random.randint(50, 3000, size=n_samples),
            "діабет": np.random.randint(0, 2, size=n_samples),
            "фракція_викиду": np.random.randint(20, 80, size=n_samples),
            "високий_тиск": np.random.randint(0, 2, size=n_samples),
            "тромбоцити": np.random.uniform(100, 500, size=n_samples),
            "сироватковий_креатинін": np.random.uniform(0.5, 5.0, size=n_samples),
            "сироватковий_натрій": np.random.randint(120, 140, size=n_samples),
            "стать": np.random.randint(0, 2, size=n_samples),
            "куріння": np.random.randint(0, 2, size=n_samples),
            "час": np.random.randint(30, 300, size=n_samples),
            "смертність": np.random.randint(0, 2, size=n_samples)
        })

    @staticmethod
    def plot_distribution(data: pd.DataFrame, feature: str, title: str) -> None:
        """Візуалізація розподілу змінної."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=feature, hue="смертність", multiple="stack")
        plt.title(title)
        plt.xlabel(feature.capitalize())
        plt.ylabel("Кількість")
        plt.show()

    @staticmethod
    def perform_logistic_regression(data: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Виконання логістичної регресії."""
        X = data.drop("смертність", axis=1)
        y = data["смертність"]
        X = sm.add_constant(X)
        return sm.Logit(y, X).fit()

    def perform_clustering(self, data: pd.DataFrame, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans]:
        """Кластеризація пацієнтів."""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.drop("смертність", axis=1))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        clusters = kmeans.fit_predict(data_scaled)
        return clusters, kmeans

    @staticmethod
    def visualize_clusters(data: pd.DataFrame, clusters: np.ndarray) -> None:
        """Візуалізація результатів кластеризації."""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.drop("смертність", axis=1))
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis')
        plt.title("Кластери пацієнтів на основі їхніх характеристик")
        plt.xlabel("Компонента 1")
        plt.ylabel("Компонента 2")
        plt.colorbar(label="Кластер")
        plt.show()


def main():
    try:
        # Ініціалізація аналізатора
        analyzer = MedicalDataAnalyzer(seed=0)
        logger.info("Початок аналізу медичних даних")

        # Генерація даних
        data = analyzer.generate_sample_data()
        logger.info(f"Згенеровано датасет з {len(data)} спостереженнями")

        # Базова статистика
        print("\nОписова статистика:")
        print(data.describe())

        # Візуалізація розподілів
        analyzer.plot_distribution(data, "вік", "Розподіл віку пацієнтів залежно від смертності")

        # Логістична регресія
        log_reg_results = analyzer.perform_logistic_regression(data)
        print("\nРезультати логістичної регресії:")
        print(log_reg_results.summary())

        # Кластеризація
        clusters, _ = analyzer.perform_clustering(data)
        analyzer.visualize_clusters(data, clusters)

        logger.info("Аналіз успішно завершено")

    except Exception as e:
        logger.error(f"Помилка під час аналізу: {str(e)}")
        raise


if __name__ == "__main__":
    main()
