
from .task import Task, ConditionTask, ScheduledTask, Inspection


class TaskFactory:

    @classmethod
    def create_task(self, *args, **kwargs):

            if task_type == 'Task':

                task = Task(*args, **kwargs)

            elif task_type == 'ConditionTask':

                task = ConditionTask(*args, **kwargs)

            elif task_type == 'ScheduledTask':

                task = ScheduledTask(*args, **kwargs)

            elif task_type == 'Inspection':

                task = Inspection(*args, **kwargs)

        return task