import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

__all__ = ['Preprocessor']


class Preprocessor:

    def __init__(
        self,
        task: str
    ) -> None:
        
        self._cat_imputer = SimpleImputer(strategy='most_frequent')
        self._num_imputer = SimpleImputer(strategy='mean')

        self._cat_encoder = OrdinalEncoder()
        self._x_scaler = StandardScaler()
        
        task1, task2 = task.split(':')
        if task2 == 'bincls' or task2 == 'cls':
            self._y_scaler = OrdinalEncoder(dtype='int')
        else:
            self._y_scaler = StandardScaler()

    def fit_transform(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        x, y = x.copy(), y.copy()

        catcols = [col for col in x.columns if x[col].dtype == 'object']
        numcols = [col for col in x.columns if x[col].dtype != 'object']
        x = x.groupby(x.index.name).ffill()

        if catcols:
            x[catcols] = self._cat_imputer.fit_transform(x[catcols].values)
            x[catcols] = self._cat_encoder.fit_transform(x[catcols].values)
        if numcols:
            for col in x.columns:
                if pd.isna(x[col]).sum() == len(x):
                    x[col] = x[col].fillna(0)
            x[numcols] = self._num_imputer.fit_transform(x[numcols].values)

        x = pd.DataFrame(self._x_scaler.fit_transform(x.values), index=x.index, columns=x.columns)
        y = pd.DataFrame(self._y_scaler.fit_transform(y.values), index=y.index, columns=y.columns)

        return x, y

    def transform(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        x, y = x.copy(), y.copy()

        catcols = [col for col in x.columns if x[col].dtype == 'object']
        numcols = [col for col in x.columns if x[col].dtype != 'object']
        x = x.groupby(x.index.name).ffill()

        if catcols:
            x[catcols] = self._cat_imputer.transform(x[catcols].values)
            x[catcols] = self._cat_encoder.transform(x[catcols].values)
        if numcols:
            x[numcols] = self._num_imputer.transform(x[numcols].values)
        
        x = pd.DataFrame(self._x_scaler.transform(x.values), index=x.index, columns=x.columns)
        y = pd.DataFrame(self._y_scaler.transform(y.values), index=y.index, columns=y.columns)

        return x, y
