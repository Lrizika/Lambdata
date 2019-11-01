import sklearn.cluster
import pandas
from typing import Tuple, Optional


class KMeansPreprocessor:
	"""
	Provides an sklearn-like interface for clustering as a preprocessing step.
	"""

	def __init__(	self,
					n_clusters: int = 100,
					by_columns: list = None,
					to_column: str = 'cluster'):
		"""
		Args:
			n_clusters (int, optional): Number of clusters to create. Defaults to 100.
			by_columns (list, optional): Columns to cluster by. Defaults to all numeric columns.
			to_column (str, optional): Column name in which to save clusters. Defaults to 'cluster'.
		"""

		self.n_clusters = n_clusters
		self.by_columns = by_columns
		self.to_column = to_column
		self.kmeans = None

	def fit(self,
			X: pandas.DataFrame,
			y: None = None):
		"""
		Computes cluster centers.
		
		Args:
			X (pandas.DataFrame): Dataframe on which to cluster.
			y (None): Dummy argument for sklearn pipelining. Never used.
		"""

		from sklearn.cluster import KMeans
		from numpy import number as npnumber

		if self.by_columns is None:
			self.by_columns = X.select_dtypes(include=[npnumber]).columns

		self.kmeans = KMeans(n_clusters=self.n_clusters)
		self.kmeans.fit(X[self.by_columns])

		return(self)

	def transform(	self,
					X: pandas.DataFrame,
					y: None = None) -> pandas.DataFrame:
		"""
		Calculates the nearest cluster center and adds the label of that cluster as a column.
		
		Args:
			X (pandas.DataFrame): Dataframe on which to cluster.
			y (None): Dummy argument for sklearn pipelining. Never used.
		
		Returns:
			pandas.DataFrame: Copy of the dataframe with an added column of cluster labels.

		Raises:
			sklearn.exceptions.NotFittedError: If `transform` is called before `fit`.
		"""

		from sklearn.exceptions import NotFittedError

		if self.kmeans is None:
			raise NotFittedError(f'This {self.__class__.__name__} instance is not fitted yet')

		output = X.copy()
		clusters = self.kmeans.predict(output[self.by_columns])
		output[self.to_column] = clusters

		return(output)

	def fit_transform(	self,
						X: pandas.DataFrame,
						y: None = None) -> pandas.DataFrame:
		"""
		Computes cluster centers and adds nearest cluster labels to the dataframe as a column.
		
		Convenience method; equivalent to calling fit(X) followed by transform(X).

		Args:
			X (pandas.DataFrame): Dataframe on which to cluster.
			y (None): Dummy argument for sklearn pipelining. Never used.
		
		Returns:
			pandas.DataFrame: Copy of the dataframe with an added column of cluster labels.
		"""

		self.fit(X)
		return(self.transform(X))


def cluster(df: pandas.DataFrame,
			n_clusters: int = 100,
			by: list = None,
			kmeans: sklearn.cluster.KMeans = None
			) -> Tuple[pandas.Series, sklearn.cluster.KMeans]:
	"""
	Clusters a dataframe by a set of columns
	
	Args:
		df (pandas.DataFrame): Datadf to cluster
		n_clusters (int, optional): Number of clusters to create. Defaults to 100.
		by (list, optional): Columns to cluster by. Defaults to all columns.
		kmeans (sklearn.cluster.KMeans, optional): If provided, fits to an existing set of clusters. Defaults to a new clustering.
	
	Returns:
		Tuple[pandas.Series, sklearn.cluster.KMeans]: Series of cluster labels and KMeans instance for future clustering.
	"""

	from sklearn.cluster import KMeans

	if by is None:
		by = df.columns

	if kmeans is None:
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(df[by])
		cluster = kmeans.labels_
	else:
		cluster = kmeans.predict(df[by])
	return(cluster, kmeans)


def oneHot(	df: pandas.DataFrame, 
			cols: Optional[list] = None,
			exclude_cols: Optional[list] = None,
			max_cardinality: Optional[int] = None
			) -> pandas.DataFrame:
	"""
	One-hot encodes the dataframe.
	
	Args:
		df (pandas.DataFrame): Datadf to clean
		cols (list, optional): Columns to one-hot encode. Defaults to all string columns.
		exclude_cols (list, optional): Columns to skip one-hot encoding. Defaults to None.
		max_cardinality (int, optional): Maximum cardinality of columns to encode. Defaults to no maximum cardinality.
	
	Returns:
		pandas.DataFrame: The one_hot_encoded dataframe.
	"""
	import category_encoders, numpy

	one_hot_encoded = df.copy()

	if cols is None: 
		cols = list(one_hot_encoded.columns[one_hot_encoded.dtypes=='object'])

	if exclude_cols is None:
		exclude_cols = []
	for col in exclude_cols:
		cols.remove(col)

	if max_cardinality is not None:
		uniques = one_hot_encoded[cols].nunique()
		cols = list(uniques[uniques <= max_cardinality].index)

	encoder = category_encoders.OneHotEncoder(return_df=True, use_cat_names=True, cols=cols)
	one_hot_encoded = encoder.fit_transform(one_hot_encoded)

	return(one_hot_encoded)


def keepTopN(	column: pandas.Series,
				n: int,
				default: Optional[object] = None
				) -> pandas.Series:
	"""
	Keeps the top n most popular values of a Series, while replacing the rest with `default`
	
	Args:
		column (pandas.Series): Series to operate on
		n (int): How many values to keep
		default (object, optional): Defaults to NaN. Value with which to replace remaining values
	
	Returns:
		pandas.Series: Series with the most popular n values
	"""
	import numpy

	if default is None: default = numpy.nan

	val_counts = column.value_counts()
	if n > len(val_counts): n = len(val_counts)
	top_n = list(val_counts[: n].index)
	return(column.where(column.isin(top_n), other=default))

