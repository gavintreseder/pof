
"""
    Filename: task_group.py
    Description: Contains the code for implementing a TaskGroup class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

if __package__ is None or __package__ == '':
    from distribution import Distribution
else:
    from pofpof.distribution import Distribution



#from pof.task import Task

class TaskGroup():
    """
            Methods:
                Che
    """
    
    def __init__(self):

        self.tasks = None
        self.triggered = None


if __name__ == "__main__":
    TaskGroup = TaskGroup()
    print("TaskGroup - OK")