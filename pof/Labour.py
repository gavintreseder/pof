


class Resource():

    def __init__(self):


        self.cost_rate = 100
        
        # Fields from Isograph
        self.id = NotImplemented
        self.type = NotImplemented
        self.number_available = NotImplemented
        self.corrective_logistic_delay = NotImplemented
        self.corrective_call_out_cost = NotImplemented
        self.scheduled_call_out_cost = NotImplemented

    def Cost(self):

        return self.cost_rate


class LabourResource(Resource):

    def __init__(self):
        self.status = NotImplemented