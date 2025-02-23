from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib


def create_model():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data_set = load_iris()

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5)),
        ('classifier', SVC(kernel='linear', C=0.5))
    ])
    param_grid = {
        'pca__n_components': [2, 3, 4],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("best parameters:", grid_search.best_params_)
    print(f'Model accuracy: {grid_search.score(X_test, y_test)}')

    joblib.dump(grid_search, 'model.pkl')
