import pandas as pd
import numpy as np
# np.random.seed(5)


class Users:
    """ User group characteristics """

    def __init__(self, city):
        assert isinstance(city, str)
        self.city_scaling = {'SiouxFalls': 0.5, 'Pigou_reprise': 1}  # scale user flows to ensure feasibility
        self.city = city
        self.raw_od = pd.read_csv("Locations/" + city + "/od.csv")
        self.num_users = None
        self.vot_mean_array = None
        self.data = self._generate_users()


    def _generate_users(self):

        splits = 1
        df = pd.DataFrame(np.repeat(self.raw_od.values, splits, axis=0), columns=self.raw_od.columns)
        self.num_users = splits * df.shape[0]
        self.vot_mean_array = 10 * np.ones(self.num_users) + 40 * np.random.rand(self.num_users)

        df['volume'] = round(self.city_scaling[self.city] * df['volume']/splits)

        df['vot'] = self.vot_realization()

        df.rename(columns={"origin": "orig", "destination": "dest", "volume": "vol"}, inplace=True)
        data = df.to_dict('index')
        return data

    def vot_list(self):
        vot = [self.data[i]['vot'] for i in range(len(self.data))]
        return vot

    def vot_array(self):
        vot = [self.data[i]['vot'] for i in range(len(self.data))]
        return np.array(vot)

    def user_flow_list(self):
        user_flow = [self.data[i]['vol'] for i in range(len(self.data))]
        return user_flow

    def vot_realization(self, fixed_vot=False):
        if fixed_vot is True:
            vot_array = np.ones(self.num_users)
        elif type(fixed_vot) is float:
            vot_array = fixed_vot * np.ones(self.num_users)
        else:
            # vot_array = 0.7 * np.ones(self.num_users) + 0.6 * np.random.rand(self.num_users)  # Med variance
            # vot_array = 1.2 * np.ones(self.num_users) + 0.6 * np.random.rand(self.num_users)  # High mean
            # vot_array = 0.2 * np.ones(self.num_users) + 0.6 * np.random.rand(self.num_users)  # Low mean
            # vot_array = 0.9 * np.ones(self.num_users) + 0.2 * np.random.rand(self.num_users)  # Low variance
            vot_array = 0.8 * self.vot_mean_array + 0.4 * self.vot_mean_array * np.random.rand(self.num_users)
        return vot_array

    def new_instance(self, fixed_vot=False):
        new_vot = self.vot_realization(fixed_vot=fixed_vot)
        for u in self.data.keys():
            self.data[u]['vot'] = new_vot[u]
        return None

    def population_vot_mean(self):
        weighted_sum = 0
        total_sum = 0
        for u in self.data.keys():
            weighted_sum += self.data[u]['vol'] * self.vot_mean_array[u]
            total_sum += self.data[u]['vol']
        return weighted_sum / total_sum
