"""
    Filename: model_loader.py
    Description: Contains the code for implementing a ModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import pandas as pd



class AssetModelLoader:
    """
    ModelLoader is used to load model parameters from an excel sheet and transform them into a json/dict structure that can be used to load pof objects.


    Usage:

    aml = AssetModelLoader()
    aml.load(filename)
    """

    def __init__(self):

        # file location

        self.filename = None
    

    def load(self, filename=None):

        # Load a filename if it has been passed to it
        if filename is None:
            if self.filename is None:
                raise Exception("No file specified")
        else:
            self.set_filename(filename)
        
        # Load the data
        #if self.filename.endswith()
        data = self.load_xlsx()

        return data


    def set_filename(self, filename):
        #TODO add some error checking around this

        self.filename = filename
    
    def load_xlsx(self):
        """
        Transform a .xlsx file into a dictionary which can be used to create an asset model
        """

        self.read_xlsx()
        comp_data = self._get_component_data(self.df)

        return comp_data

    def read_xlsx(self):
        """
        Transform a .xlsx file into a dictionary which can be used to create an asset model
        """

        df = pd.read_excel(self.filename, sheet_name='Model Input', header=[0,1,2])

        # Create keys
        keys = [
            ('asset_model', 'component', 'name'),
            ('failure_model', 'failure_mode', 'name'),
            ('task_model', 'task', 'name'),
        ]

        # Drop rows with no data
        df = df.dropna(axis=1, how='all')

        # Propogate keys for 1 to many relationships
        df[keys] = df[keys].ffill()

        self.df = df

    def load_pof_object(self):


        NotImplemented


    def to_xls(self):


        NotImplemented

    # ********************* validate methods *****************************

    def _validate_keys(self, keys, df):
        missing_keys = [key for key in keys if key not in df.columns]

        if bool(missing_keys):
            print("Missing Keys: %s" %(missing_keys))
            return False
        else:
            return True


    # ********************* excel load methods *****************************

    def _get_component_data(self, df):
        comps_data = dict()

        # Get the Component information
        comp_key = ('asset_model', 'component', 'name')
        components = df[comp_key].dropna().unique()

        df_comps = df[['asset_model', 'failure_model', 'condition_model', 'task_model', 'trigger_model', 'impact_model']].set_index(comp_key)

        for comp in components:

            df_comp = df_comps.loc[[comp]]

            # Get the FailureMode information
            fm_data = self._get_failure_mode_data(df_comp)

            comp_data = dict(
                fm = fm_data,
            )

            comps_data.update({
                comp : comp_data
            })

        return comps_data

    def _get_failure_mode_data(self, df_comp):

        fms_data = dict() #TODO
        fm_key = ('failure_model', 'failure_mode', 'name')
        failure_modes = df_comp[fm_key].unique()
        df_fms = df_comp[['failure_model', 'condition_model', 'task_model', 'trigger_model', 'impact_model']].set_index(fm_key)

        for fm in failure_modes:

            df_fm = df_fms.loc[[fm]]

            # Get the Task information
            tasks_data = self._get_task_data(df_fm)
            
            # Get the Distribution information
            dist_data = self._get_dist_data(df_fm)

            # Get the Condition information
            condition_data = self._get_condition_data(df_fm)

            fm_data = dict(
                name = fm,
                conditions = condition_data,
                tasks = tasks_data,
                untreated = dist_data,
            )
            fms_data.update({
                fm : fm_data,
            })

        return fms_data

    def _get_condition_data(self, df_fm):
        #TODO update for new arrangement
        df_cond = df_fm['condition_model']
        conditions = df_cond['condition']['name'].dropna().to_numpy()
        cond_data = {cond : None for cond in conditions}
        return cond_data

    def _get_dist_data(self,df_fm):

        df_dist = df_fm['failure_model'].dropna()
        df_dist.columns = df_dist.columns.droplevel()

        try:
            dist_data = df_dist.iloc[0].dropna().to_dict()
        except IndexError:
            dist_data = df_dist.dropna().to_dict()

        return dist_data

    def _get_task_data(self, df_fm):
        """Takes a dataframe for a failure mode and returns a dict of task data
        """
        tasks_data = dict()
        task_key = ('task_model', 'task', 'name')
        tasks = df_fm[task_key].unique()
        df_tasks = df_fm[['task_model', 'trigger_model', 'impact_model']].set_index(task_key)

        for task in tasks:

            df_task = df_tasks.loc[[task]].dropna(axis=0, how='all')

            # Trigger information
            try:
                state = df_task['trigger_model']['state'].iloc[0].dropna().to_dict(),
            except:
                state = df_task['trigger_model']['state'].dropna().to_dict(),

            trigger_data = dict(
                state = state,
                condition = df_task['trigger_model']['condition'].dropna().set_index('name').to_dict('index')
            )

            # Impact information
            try:
                state = df_task['impact_model']['state'].iloc[0].dropna().to_dict(),
            except:
                state = df_task['impact_model']['state'].dropna().to_dict(),
                
            impact_data = dict(
                state = state,
                condition = df_task['impact_model']['condition'].dropna().set_index('name').to_dict('index')
            )

            # Tasks specific information
            df_tsi = df_task[('task_model')].dropna(how='all').dropna(axis=1)
            df_tsi.columns = df_tsi.columns.droplevel()

            task_data = df_tsi.to_dict('index') #TODO currently has too many vars
            try:
                task_data[task].update(dict(
                    name=task,
                    triggers=trigger_data,
                    impacts = impact_data,
                ))
            except:
                task_data = dict(
                    task = dict(
                        name=task,
                        triggers=trigger_data,
                        impacts = impact_data,
                    )
                )

            tasks_data.update(task_data)

        return tasks_data