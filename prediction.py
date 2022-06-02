import numpy as np
import pandas as pd
from plot_analysis import preprocess
from scipy.stats import norm
from scipy.stats import expon

class IntervalPricePrediction:
    def __init__(self):
        self.df = self.preprocess_data()
        self.model_params = self.estimate_parameters()
    
    def preprocess_data(self):
        """Reads the excel file and returns a preprocessed dataframe.

        This will drop all rows that do not have a value under the 'model' category
        and obtains only the top 250 models in terms of frequency.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """    
        df = preprocess('vehicles_cleaned_imputed.csv')
        df.dropna(axis='index', subset=['model'], inplace=True)

        # # Ordinal encoding
        # title_map = {'missing': 0, 'parts only': 1, 'salvage': 2, 'lien': 3, 'rebuilt': 4, 'clean': 5}
        # df['title_status'] = df['title_status'].map(title_map)
        # size_map = {'sub-compact': 0, 'compact': 1, 'mid-size': 2, 'full-size': 3}
        # df['size'] = df['size'].map(size_map)

        # Consider only top 250 frequent models
        df = df[df['model'].isin(df.groupby('model')['model'].count().sort_values(ascending=False)[:250].index)]
        return df

    def estimate_parameters(self):
        """Estimates the parameters of the distributions.
        Price given car model_name is modelled as a Gaussian and mean and variance are calculated using MLE.
        Time of next post given car model_name is modelled as an exponential distribution and lambda is estimated using MLE.

        Returns:
            pd.DataFrame: Dataframe containing model parameters for each car model_name.
        """    
        model_params = self.df.groupby('model').aggregate({"price": ['mean','std'], "model": 'count', "posting_date":['max','min']})
        return model_params

    def get_CI(self, p1:float, p2:float, alpha=0.95):
        """Calculates the probability of a model being in the desired price range and the 95% interval for the next post.

        Args:
            p1 (float): lower bound of desired price
            p2 (float): upper bound of desired price
            alpha (float): Confidence interval value

        Returns:
            list (tuple): [(manufacturer, model_name, probability of price in [p1,p2] for model, lower bound of time in mins, upper bound of time in mins)]
        """ 
        # P(model_name)
        n = self.model_params['model']['count']
        p = n/self.model_params['model']['count'].sum() 
        # Mean and std of the Gaussian model      
        mean = self.model_params['price']['mean']
        std = self.model_params['price']['std']/np.sqrt(n)
        # lambdas of the exponential distribution
        lambdas = n/(self.model_params['posting_date']['max'] - self.model_params['posting_date']['min']).astype('timedelta64[m]')
        # P(price in [p1,p2] | model_name)
        self.model_params['cond_prob'] = norm.cdf(p2, loc=mean, scale=std) - norm.cdf(p1, loc=mean, scale=std)
        # P(price in [p1,p2], model_name)
        joint_prob = self.model_params['cond_prob']*p
        # consider only models with high probability of being in desired price range
        high_prob_models = joint_prob.sort_values(ascending=False)[:10].index
        # T(price_range, model_name) is exponential with lambda defined below
        lambda_joint = lambdas.loc[high_prob_models]/self.model_params.loc[high_prob_models]['cond_prob']
        return list(zip(self.df[self.df['model'].isin(high_prob_models)]['manufacturer'], high_prob_models,
                        joint_prob.sort_values(ascending=False)[:10],
                        expon.interval(alpha, scale=1/lambda_joint)[0],
                        expon.interval(alpha, scale=1/lambda_joint)[1]))
