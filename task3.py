import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FoodProductAnalyzer:
    """Клас для аналізу та сегментації харчових продуктів."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        np.random.seed(random_state)
        self.clf = None
        self.kmeans = None
        self.scaler = StandardScaler()

    def generate_sample_data(self, size: int = 300) -> pd.DataFrame:
        """Генерація тестового набору даних про харчові продукти."""

        return pd.DataFrame({
            'Продукт': np.random.choice(['Хліб', 'Молоко', 'Сир', 'Йогурт', 'М\'ясо'], size=size),
            'Ціна': np.random.uniform(10.0, 200.0, size=size),
            'Вага_кг': np.random.uniform(0.1, 5.0, size=size),
            'Категорія': np.random.choice(['Молочні', 'Хлібобулочні', 'М\'ясні'], size=size),
            'Термін_придатності_днів': np.random.randint(1, 30, size=size),
            'Прибуток': np.random.uniform(-20, 100, size=size),
            'Знижка': np.random.uniform(0, 0.3, size=size),
            'Калорійність': np.random.uniform(50, 400, size=size),
            'Білки': np.random.uniform(2, 30, size=size),
            'Жири': np.random.uniform(0, 40, size=size),
            'Вуглеводи': np.random.uniform(0, 50, size=size)
        })

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Підготовка даних для аналізу харчових продуктів."""

        # Створення копії для уникнення змін оригінальних даних
        df = data.copy()

        # Розрахунок додаткових характеристик
        df['Ціна_за_кг'] = df['Ціна'] / df['Вага_кг']
        df['Енергетична_цінність'] = df['Білки'] * 4 + df['Жири'] * 9 + df['Вуглеводи'] * 4

        # Конвертація категоріальних змінних
        df = pd.get_dummies(df, columns=['Продукт', 'Категорія'], drop_first=True)

        # Створення цільової змінної (прибутковість продукту)
        df['Прибутковий'] = (df['Прибуток'] > df['Прибуток'].median()).astype(int)

        # Розділення на ознаки та цільову змінну
        features = ['Ціна', 'Вага_кг', 'Термін_придатності_днів', 'Ціна_за_кг',
                    'Калорійність', 'Білки', 'Жири', 'Вуглеводи', 'Енергетична_цінність'] + \
                   [col for col in df.columns if col.startswith(('Продукт_', 'Категорія_'))]

        X = df[features]
        y = df['Прибутковий']

        return X, y

    def train_classifier(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> Dict[str, Any]:
        """Навчання класифікатора для передбачення прибутковості продуктів."""

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.clf = RandomForestClassifier(random_state=self.random_state)
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)

        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.clf.feature_importances_))
        }

    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """Кластеризація харчових продуктів."""

        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = self.kmeans.fit_predict(X_scaled)

        return clusters

    def analyze_clusters(self, data: pd.DataFrame, clusters: np.ndarray) -> Dict[str, pd.DataFrame]:
        """Аналіз характеристик кластерів."""

        df = data.copy()
        df['Кластер'] = clusters

        cluster_stats = {}
        for cluster in range(len(np.unique(clusters))):
            cluster_data = df[df['Кластер'] == cluster]
            stats = cluster_data.describe()
            cluster_stats[f'Кластер_{cluster}'] = stats

        return cluster_stats

    def visualize_clusters(self, X: pd.DataFrame, clusters: np.ndarray) -> None:
        """Візуалізація результатів кластеризації."""

        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.xlabel('Перша головна компонента')
        plt.ylabel('Друга головна компонента')
        plt.title('Кластеризація харчових продуктів')
        plt.colorbar(scatter, label='Кластер')
        plt.show()

    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """Візуалізація важливості характеристик продуктів."""

        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['importance'])
        importance_df.sort_values('importance', ascending=True).tail(10).plot(kind='barh')
        plt.title('Топ-10 важливих характеристик продуктів')
        plt.xlabel('Важливість')
        plt.tight_layout()
        plt.show()


def main():
    try:
        logger.info("Початок аналізу харчових продуктів")

        # Ініціалізація аналізатора
        analyzer = FoodProductAnalyzer()

        # Генерація даних
        data = analyzer.generate_sample_data()
        logger.info(f"Згенеровано датасет з {len(data)} записами")

        # Підготовка даних
        X, y = analyzer.preprocess_data(data)
        logger.info("Дані підготовлено для аналізу")

        # Класифікація
        classification_results = analyzer.train_classifier(X, y)
        print("\nРезультати класифікації прибутковості:")
        print(classification_results['classification_report'])

        # Візуалізація важливості характеристик
        analyzer.plot_feature_importance(classification_results['feature_importance'])

        # Кластеризація
        clusters = analyzer.perform_clustering(X)

        # Аналіз кластерів
        cluster_stats = analyzer.analyze_clusters(data, clusters)
        print("\nСтатистика по кластерам:")
        for cluster_name, stats in cluster_stats.items():
            print(f"\n{cluster_name}:")
            print(stats)

        # Візуалізація кластерів
        analyzer.visualize_clusters(X, clusters)

        logger.info("Аналіз успішно завершено")

    except Exception as e:
        logger.error(f"Помилка під час аналізу: {str(e)}")
        raise


if __name__ == "__main__":
    main()