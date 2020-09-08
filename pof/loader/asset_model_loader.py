"""
    Filename: model_loader.py
    Description: Contains the code for implementing a ModelLoader class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

# Read a file

# Read 



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
        self.load_xlsx()


    def set_filename(self, filename):
        #TODO add some error checking around this

        self.filename = filename
    
    def load_xlsx(self):
        """
        Transform a .xlsx file into a dictionary which can be used to create an asset model
        """
        


        NotImplemented

    def load_pof_object(self):


        NotImplemented


    def to_xls(self):


        NotImplemented


    # ********************* excel load methods *****************************

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
        trigger_data = dict(
            state = df_task['trigger_model']['state'].iloc[0].dropna().to_dict(),
            condition = df_task['trigger_model']['condition'].dropna().set_index('name').to_dict('index')
        )

        # Impact information
        impact_data = dict(
            state = df_task['impact_model']['state'].iloc[0].dropna().to_dict(),
            condition = df_task['impact_model']['condition'].dropna().set_index('name').to_dict('index')
        )

        # Tasks specific information
        df_tsi = df_task[('task_model')].dropna(how='all').dropna(axis=1)
        df_tsi.columns = df_tsi.columns.droplevel()

        task_data = df_tsi.to_dict('index') #TODO currently has too many vars
        task_data[task].update(dict(
            name=task,
            triggers=trigger_data,
            impacts = impact_data,
        ))

        tasks_data.update(task_data)

    return tasks_data