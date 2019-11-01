"""
Pytest unit tests for pandas_utils
"""

from lambdata_lrizika import pandas_utils
import pandas
import numpy

example_dataframe = pandas.DataFrame({'column_low': [0,1,1,2,2,3,4,5,6],
									'column_med': [2,4,3,1,4,6,2,1,4],
									'column_high': [10,11,33,11,6,14,21,3,10],
									'column_catnum': ['2','5','2','3','5','1','3','1','0'],
									'column_catstr': ['one','alpha','a','a','a','george','alpha','seventeen','alpha']})

class TestKeepTopN:
	"""
	Tests lambdata_lrizika.pandas_utils.keepTopN
	"""

	def test_num_all(self):
		""" Tests on a numeric column while keeping all values """
		assert pandas_utils.keepTopN(example_dataframe['column_low'], 99).equals(pandas.Series([0,1,1,2,2,3,4,5,6]))
	def test_num_2_with_def_num(self):
		""" Tests on a numeric column while keeping two values and using a numeric default """
		assert pandas_utils.keepTopN(example_dataframe['column_low'], 2, default=-1).equals(pandas.Series([-1,1,1,2,2,-1,-1,-1,-1]))
	def test_num_2_with_def_str(self):
		""" Tests on a numeric column while keeping two values and using a string default """
		assert pandas_utils.keepTopN(example_dataframe['column_low'], 2, default='string').equals(pandas.Series(['string',1,1,2,2,'string','string','string','string']))
	def test_num_2(self):
		""" Tests on a numeric column while keeping two values """
		assert pandas_utils.keepTopN(example_dataframe['column_low'], 2).equals(pandas.Series([numpy.nan,1,1,2,2,numpy.nan,numpy.nan,numpy.nan,numpy.nan]))
	def test_num_none(self):
		""" Tests on a numeric column while keeping no values """
		assert pandas_utils.keepTopN(example_dataframe['column_low'], 0).equals(pandas.Series([numpy.nan]*9))
	def test_str_all(self):
		""" Tests on a string column while keeping all values """
		assert pandas_utils.keepTopN(example_dataframe['column_catstr'], 99).equals(pandas.Series(['one','alpha','a','a','a','george','alpha','seventeen','alpha']))
	def test_str_2(self):
		""" Tests on a string column while keeping two values """
		assert pandas_utils.keepTopN(example_dataframe['column_catstr'], 2).equals(pandas.Series([numpy.nan,'alpha','a','a','a',numpy.nan,'alpha',numpy.nan,'alpha']))
	def test_str_2_with_def_num(self):
		""" Tests on a string column while keeping two values and using a numeric default """
		assert pandas_utils.keepTopN(example_dataframe['column_catstr'], 2, default=1).equals(pandas.Series([1,'alpha','a','a','a',1,'alpha',1,'alpha']))
	def test_str_none(self):
		""" Tests on a string column while keeping no values """
		assert pandas_utils.keepTopN(example_dataframe['column_catstr'], 0).equals(pandas.Series([numpy.nan]*9, dtype='object'))

class TestOneHot:
	"""
	Tests lambdata_lrizika.pandas_utils.oneHot
	"""

	expected_default = pandas.DataFrame(
		{'column_low': [0, 1, 1, 2, 2, 3, 4, 5, 6],
		'column_med': [2, 4, 3, 1, 4, 6, 2, 1, 4],
		'column_high': [10, 11, 33, 11, 6, 14, 21, 3, 10],
		'column_catnum_2': [1, 0, 1, 0, 0, 0, 0, 0, 0],
		'column_catnum_5': [0, 1, 0, 0, 1, 0, 0, 0, 0],
		'column_catnum_3': [0, 0, 0, 1, 0, 0, 1, 0, 0],
		'column_catnum_1': [0, 0, 0, 0, 0, 1, 0, 1, 0],
		'column_catnum_0': [0, 0, 0, 0, 0, 0, 0, 0, 1],
		'column_catstr_one': [1, 0, 0, 0, 0, 0, 0, 0, 0],
		'column_catstr_alpha': [0, 1, 0, 0, 0, 0, 1, 0, 1],
		'column_catstr_a': [0, 0, 1, 1, 1, 0, 0, 0, 0],
		'column_catstr_george': [0, 0, 0, 0, 0, 1, 0, 0, 0],
		'column_catstr_seventeen': [0, 0, 0, 0, 0, 0, 0, 1, 0]}
	)

	def test_default(self):
		""" Tests with the default parameters """
		assert pandas_utils.oneHot(example_dataframe).equals(self.expected_default)
	def test_default_explicit(self):
		""" Tests with the default parameters explicitly invoked """
		assert pandas_utils.oneHot(	example_dataframe,
									cols=['column_catnum', 'column_catstr'],
									exclude_cols=None,
									max_cardinality=None).equals(self.expected_default)
	def test_max_cardinality(self):
		""" Tests that a low max cardinality results in no encoding """
		assert pandas_utils.oneHot(	example_dataframe,
									max_cardinality=2).equals(example_dataframe)

class TestKMeansPreprocessor:
	"""
	Tests lambdata_lrizika.pandas_utils.KMeansPreprocessor
	"""

	def test_fit_transform(self):
		""" Tests that fit_transform results in expected output format """
		clusterer_3 = pandas_utils.KMeansPreprocessor(n_clusters=3)
		clustered = clusterer_3.fit_transform(example_dataframe)
		assert list(clustered.columns) == list(example_dataframe.columns) + ['cluster']
	def test_fit_transform_results(self):
		""" Tests that clustering with two clusters creates expected results """
		clusterer_2 = pandas_utils.KMeansPreprocessor(n_clusters=2)
		clustered = clusterer_2.fit_transform(example_dataframe)
		assert (clustered['cluster'] == clustered.iloc[0]['cluster']).equals(pandas.Series([True, True, False, True, True, True, False, True, True]))
		assert (clustered['cluster'] == clustered.iloc[2]['cluster']).equals(pandas.Series([False, False, True, False, False, False, True, False, False]))
	def test_pipeline(self):
		""" Tests that the preprocessor works in a pipeline """
		clusterer_3 = pandas_utils.KMeansPreprocessor(n_clusters=3)
		from sklearn.pipeline import Pipeline
		pipeline = Pipeline([('Clusterer', clusterer_3)])
		assert set(pipeline.fit_transform(example_dataframe)['cluster'].unique()) == {0,1,2}
		assert pipeline.fit_transform(example_dataframe).shape == clusterer_3.fit_transform(example_dataframe).shape


